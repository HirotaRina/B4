import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import torch


max_classes_num = 200
## sc network
class SC(nn.Module):
    def __init__(self, word_embed_size, hidden_size, vocab_size, num_classes,
            dropout_prob=0.5):
        super(SC, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)

        lstm1_input_size = word_embed_size

        self.lstm = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
                #bidirectional=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.init_weights()

        self.input_size = vocab_size
        self.output_size = num_classes
        self.dropout_prob = dropout_prob

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, captions, lengths):
        embeddings = self.word_embed(captions)
        embeddings = F.dropout(embeddings, p=self.dropout_prob, training=self.training)

        packed = embeddings
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        #hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), hiddens.size(2))
        idx = idx.unsqueeze(1)
        idx = idx.to(hiddens.device)

        # Shape: (batch_size, hidden_size)
        last_hiddens = hiddens.gather(1, idx).squeeze(1)

        last_hiddens = F.dropout(last_hiddens, p=self.dropout_prob, training=self.training)
        outputs = self.linear(last_hiddens)
        return outputs



# cub
from loader.data.data_prep import DataPreparation
from loader.misc import get_split_str

# Data parameters
data_folder = '/mnt/exthd1/coco2014/'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

def cubdata_load():
    ## cub data
    cub_train = True
    cub_eval_ckpt = None
    cub_dataset = "cub"
    cub_data_path = "./data"
    cub_pretrained_model = "vgg16"
    cub_batch_size = batch_size
    cub_num_workers = workers
    split = get_split_str(cub_train, bool(cub_eval_ckpt), cub_dataset)
    data_prep = DataPreparation(cub_dataset, cub_data_path)
    
    ##
    dataset, data_loader = data_prep.get_dataset_and_loader(split, cub_pretrained_model, batch_size=cub_batch_size, num_workers=cub_num_workers)
    dataset.set_label_usage(True)

    cub_train = False
    split = get_split_str(cub_train, bool(cub_eval_ckpt), cub_dataset)
    val_dataset, val_data_loader = data_prep.get_dataset_and_loader(split, cub_pretrained_model, batch_size=cub_batch_size, num_workers=cub_num_workers)
    val_dataset.set_label_usage(True)
#    vocab = dataset.vocab
    return dataset, data_loader, val_dataset, val_data_loader
    

def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
#    print(cub_word_map["a"]) -> 4
#    print(word_map["a"]) -> 1

    #########################################
    #########################################
    # load cub data
    dataset, data_loader, val_dataset, val_data_loader, = cubdata_load()

    #for i, (image_input, word_inputs, word_targets, lengths, ids, captions, labels, allcaps, *excess) in enumerate(data_loader):
    #  print(image_input.shape)
    #  print(image_input[0]) # image
    #  print(captions.shape)
    #  print(captions[0])
    #  print(lengths)
    #  print(labels[0])
    #  print(ids)
    #  print(allcaps.shape)
    #  print(allcaps[0])
    #  break

    train_loader = data_loader
    val_loader = val_data_loader
    vocab = val_dataset.vocab
    vocab_length = vocab.__len__()

    # create wordmap
    cub_word_map = {}
    for i in range(vocab_length):
      cub_word_map[vocab.get_word_from_idx(i)] = i
    word_map = cub_word_map
 
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #####################
    #####################
    ## reinforce sc, sinuhodo OSOI
    #####################
    #####################
    sc_model = SC(256, 256, len(dataset.vocab), 200).to(device)
    sc_optimizer = torch.optim.Adam(sc_model.parameters(), lr=0.005)
    sc_criterion = nn.CrossEntropyLoss().to(device)
    sc_model.load_state_dict(torch.load('./sc.model.pth.tar'))
    sc_model.train()
    for epoch in range(1):
        for i, (imgs, word_inputs, word_targets, caplens, caplens_raw, ids, caps, caps_fit, labels, img_id, allcaps, *excess) in enumerate(train_loader):
            caps = caps.to(device)
            labels = labels.to(device)
            sc_optimizer.zero_grad()
            outputs = sc_model(caps, caplens_raw)
            loss = sc_criterion(outputs, labels)
            loss.backward()
            sc_optimizer.step()
            #
            if(i%100 == 0):
                _, est = torch.max(outputs.data, dim=1)
                sc_total = labels.size(0)
                sc_correct = (est == labels).sum().item()
                print(est)
                print(labels)
                print("accuracy: " + str(sc_correct/sc_total))
                print(loss.data.item())
                print("---------")
            break
        torch.save(sc_model.state_dict(), './sc.model.pth.tar')

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              sc_model=sc_model,
              sc_criterion=sc_criterion)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, sc_model, sc_criterion):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    hypotheses = list()  # hypotheses (predictions)
    sc_model.eval()

    start = time.time()
    # Batches
    for i, (imgs, word_inputs, word_targets, caplens, caplens_raw, ids, caps, caps_fit, labels, img_id, allcaps, *excess) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        labels = labels.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # Reinforce loss
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        preds_inputs = np.zeros((len(scores_copy), 70))
        for j, p in enumerate(preds):
          preds_inputs[j, :decode_lengths[j]] = np.asarray(preds[j][:decode_lengths[j]])  # remove pads
        preds_inputs = torch.from_numpy(preds_inputs).long().to(device)
        sc_outputs = sc_model(preds_inputs, decode_lengths)        # scmodel
        sc_loss = sc_criterion(sc_outputs, labels)
        loss += sc_loss.data * 0.01 # reinforce
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, word_inputs, word_targets, caplens, caplens_raw, ids, caps, caps_fit, labels, img_id, allcaps, *excess) in enumerate(val_loader):
    #    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<SOS>'], word_map['<PAD>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
