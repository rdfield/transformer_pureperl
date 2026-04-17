use Modern::Perl;
use lib '.';
use ML::Transformer;
use ML::InputEmbeddings;
use ML::PositionalEmbeddings;
use ML::Encoder;
use ML::Decoder;
use ML::Linear;
use ML::Util qw(print_2d_array create_src_mask create_tgt_mask print_1d_array softmax);
use ML::DataLoader;
use Time::HiRes qw(tv_interval gettimeofday);
use JSON;

sub train_one_epoch {
   my %args = @_;

   my $total_loss = 0;
   my $start_time = [ gettimeofday ];
   $args{dataloader}->init_iter(batch_size => $args{batch_size});

   my $start_batch = [ gettimeofday ];
   foreach my $batchno ( 0 .. $args{dataloader}->{batches} - 1) {
      my $batch = $args{dataloader}->next_batch();
      my $src_ids = $batch->{src_ids};
      my $decoder_input_ids = $batch->{decoder_input_ids};
      my $label_ids = $batch->{label_ids};
      # Create masks
      my $src_mask = create_src_mask(src_ids => $src_ids, pad_id => $args{pad_id});
      #print_2d_array("src_mask", $src_mask);
      my $tgt_mask = create_tgt_mask(tgt_ids => $decoder_input_ids, pad_id => $args{pad_id});
      my $encoder_output = $args{model}->encode(src => $src_ids, src_mask => $src_mask);
      my $decoder_output = $args{model}->decode(tgt => $decoder_input_ids, encoder_output => $encoder_output, src_mask => $src_mask, tgt_mask => $tgt_mask);
      my $final_output = $args{model}->project( batch =>  $decoder_output );
      my $seq_len = scalar(@{$final_output->[0]});
      my $vocab_size = scalar(@{$final_output->[0][0]});
      my $resized_final_output = [];
      foreach my $b ( 0 .. $args{dataloader}->batch_size() - 1) {
         foreach my $s ( 0 .. $seq_len - 1) {
            foreach my $v (0 .. $vocab_size - 1) {
               $resized_final_output->[$b * $seq_len + $s][$v] = $final_output->[$b][$s][$v];
            }
         }
      }
#      print_2d_array("resized final output", $resized_final_output);
      my $softmax = softmax($resized_final_output, 1);
#      print_2d_array("softmax by row", $softmax);
      my $resized_label_ids = [];
      foreach my $b (0 .. $args{dataloader}->batch_size() - 1) {
         foreach my $s ( 0 .. $seq_len - 1) {
            $resized_label_ids->[$b * $seq_len + $s] = $label_ids->[$b][$s];
         }
      }
#      print_1d_array("resized label ids", $resized_label_ids);
      my $loss = 0;
      my $n_tok      = scalar grep { $_ != $args{pad_id} } @$resized_label_ids;
      my $loss_scale = $n_tok || 1;                        # loss normalised by non-PAD count
      my $grad_scale = $args{batch_size} * $seq_len;       # gradient scale unchanged (keeps Adam stable)
      my $loss_gradient = [];
      foreach my $i ( 0 .. ($args{batch_size} * $seq_len) - 1) {
         if ($resized_label_ids->[$i] == $args{pad_id}) {
            $loss_gradient->[$i][$_] = 0 for 0 .. $vocab_size - 1;  # no signal from PAD
            next;
         }
         foreach my $j ( 0 .. $vocab_size - 1) {
            if ($j == $resized_label_ids->[$i]) {
               $loss += -log( $softmax->[$i][$j] ) / $loss_scale;
               $loss_gradient->[$i][$j] = ($softmax->[$i][$j] - 1) / $grad_scale;
            } else {
               $loss_gradient->[$i][$j] = $softmax->[$i][$j] / $grad_scale;
            }
         }
      }
      $total_loss += $loss;
      if (($batchno + 1) % 10 == 0) {
         my $batch_duration = sprintf("%.3f", tv_interval( $start_batch, [ gettimeofday ]));
         $start_batch = [ gettimeofday ];
         my $batch_loss = "Loss: " . sprintf("%.3f", $total_loss / ($batchno + 1)) . " Batch: " . ($batchno+1) . "/" . $args{dataloader}->{batches} . " Elapsed: $batch_duration sec";
         say $batch_loss;
      }
      # resize the loss gradient to match the original final output
      my $gradient = [];
      foreach my $b (0 .. $args{batch_size} - 1 ){
         foreach my $s (0 .. $seq_len - 1) {
            foreach my $v (0 .. $vocab_size - 1) {
               $gradient->[$b][$s][$v] = $loss_gradient->[$b * $seq_len + $s][$v];
   #            say "gradient->[$b][$s][$v] (" .  $gradient->[$b][$s][$v] . " = loss_gradient->[$b * $seq_len + $s][$v] (" . $loss_gradient->[$b * $seq_len + $s][$v] . ")";
            }
         }
         #print_2d_array("gradient $b", $gradient->[$b]);
      }
      $args{model}->update( projection => 1, decode => 1, encode => 1, gradient => $gradient, learning_rate => $args{learning_rate} );

      
   }
   my $avg_loss = sprintf("%.3f", $total_loss / $args{dataloader}->{batches});
   return $avg_loss;
   
}
    
##################################################
#### set up vocab
##################################################

# Define special tokens and vocabulary
my $PAD_TOKEN = '[PAD]';
my $SOS_TOKEN = '[SOS]'; # Start Of Sequence
my $EOS_TOKEN = '[EOS]'; # End Of Sequence

# Vocabulary for digits 0-9 plus special tokens
my $VOCAB = [$PAD_TOKEN, $SOS_TOKEN, $EOS_TOKEN];
foreach my $i (0 .. 9) {
   push @$VOCAB, "$i"
}

# Create token-to-ID and ID-to-token mappings

my ($token_to_id, $id_to_token);

foreach my $i (0 .. $#$VOCAB) {
   $id_to_token->{$i} = $VOCAB->[$i];
   $token_to_id->{$VOCAB->[$i]} = $i;
}

my $VOCAB_SIZE = scalar(@$VOCAB);
my $PAD_ID = $token_to_id->{$PAD_TOKEN};
my $SOS_ID = $token_to_id->{$SOS_TOKEN};
my $EOS_ID = $token_to_id->{$EOS_TOKEN};

say "Vocabulary: " . join(", ", @$VOCAB);
say "Vocab Size: $VOCAB_SIZE";
say "Token to ID: " . join(", ", map { $_ . " => " . $token_to_id->{$_} } keys %$token_to_id );
say "ID to Token: " . join(", ", map { $_ . " => " . $id_to_token->{$_} } keys %$id_to_token );
say "PAD ID: $PAD_ID";

sub generate_copy_task_data {
   my %args = @_;
#(num_examples: int, min_len: int, max_len: int, vocab: list):
   my $data = [];
# Exclude special tokens from the sequence content
   my $content_vocab = [];
   foreach my $token (@{$args{vocab}}) {
      next if grep { $token eq $_ } ($PAD_TOKEN, $SOS_TOKEN, $EOS_TOKEN);
      push @$content_vocab, $token;
   }
   my $max_gen_len = 0;
   my $min_gen_len = 10;
   foreach my $i (1 .. $args{num_samples}) {
      my $seq_len =  $args{min_len} + int(rand($args{max_len} - $args{min_len} + 1));
      my $sequence;
      foreach my $i (1 .. $seq_len) {
         push @$sequence, $content_vocab->[ int(rand(scalar(@$content_vocab) )) ];
      }
      # Source and Target are the same, wrapped with SOS/EOS
      my @src = ($SOS_TOKEN, @$sequence, $EOS_TOKEN);
      my @tgt = ($SOS_TOKEN, @$sequence, $EOS_TOKEN);
      push @$data, {'src' => \@src, 'tgt' =>  \@tgt};
    }

    return $data
}

################################################################
### end of vocab initialisation
################################################################
      
my $src_vocab_size = $VOCAB_SIZE;#00 # Example source vocab size
my $tgt_vocab_size = $VOCAB_SIZE;#00 # Example target vocab size


my $NUM_EXAMPLES = 1000;#0;
my $BATCH_SIZE = 24;
my $d_model = 256;#512
my $num_layers = 2; # N=6
my $num_heads = 4;#8
my $dropout = 0.1;
my $NUM_EPOCHS = 15;
my $d_ff = 1024;#2048
my $LEARNING_RATE  = 0.0001;  # to match pytorch version - also try 0.0003: 0.01 definitely overshoots, looks like 0.001 isn't great either
my $WARMUP_EPOCHS  = 3;      # linear ramp from LR/3 to LR over first 3 epochs


# Example generation
my $MIN_SEQ_LEN = 5;
my $MAX_SEQ_LEN = 10; # Keep sequences relatively short for faster training demo
srand(1);
my $raw_data = generate_copy_task_data(num_samples => $NUM_EXAMPLES, min_len => $MIN_SEQ_LEN, max_len => $MAX_SEQ_LEN, vocab => $VOCAB);
srand(99);
my $test_data = generate_copy_task_data(num_samples => 200, min_len => $MIN_SEQ_LEN, max_len => $MAX_SEQ_LEN, vocab => $VOCAB);
#say "Generated " . scalar(@$raw_data) . " examples. First example:";
#say "src: " . join(", ", @{$raw_data->[0]{src}});
#say "tgt: " . join(", ", @{$raw_data->[0]{tgt}});

my $MAX_PADDED_LEN = $MAX_SEQ_LEN + 2; # +2 for SOS and EOS

my $max_seq_len = $MAX_PADDED_LEN;#100; # Max sequence length for positional encoding

my $src_embed = ML::InputEmbeddings->new(embeddings => $d_model, vocab_size => $src_vocab_size);
my $tgt_embed = ML::InputEmbeddings->new(embeddings => $d_model, vocab_size => $tgt_vocab_size);

my $src_pos = ML::PositionalEmbeddings->new(embeddings => $d_model, max_seq_length => $max_seq_len, dropout => $dropout);
my $tgt_pos = ML::PositionalEmbeddings->new(embeddings => $d_model, max_seq_length => $max_seq_len, dropout => $dropout);

my $encoder = ML::Encoder->new( embeddings => $d_model, 
                                layers => $num_layers, 
                                max_seq_length => $max_seq_len, 
                                dropout => $dropout,
                                num_heads => $num_heads,
                                d_ff => $d_ff);

my $decoder = ML::Decoder->new( embeddings => $d_model,
                                layers => $num_layers,
                                max_seq_length => $max_seq_len,
                                dropout => $dropout,
                                d_ff => $d_ff,
                                tgt_seq_length => $max_seq_len,
                                num_heads => $num_heads );

# Projection Layer
my $projection = ML::Linear->new(insize => $d_model, outsize => $tgt_vocab_size);
$projection->set_weights_and_biases();

# --- Instantiate the Full Transformer ---
my $transformer_model = ML::Transformer->new(
    encoder=>$encoder,
    decoder=>$decoder,
    src_embed=>$src_embed,
    tgt_embed=>$tgt_embed,
    src_pos=>$src_pos,
    tgt_pos=>$tgt_pos,
    projection_layer=>$projection
);

sub warmup_lr {
    my ($epoch, $lr, $warmup) = @_;
    return $lr * $epoch / $warmup if $epoch <= $warmup;
    return $lr;
}

sub tokenize_pad {
    my ($tokens) = @_;
    my @ids = map { $token_to_id->{$_} } @$tokens;
    while (@ids < $MAX_PADDED_LEN) { push @ids, $PAD_ID }
    return [@ids[0 .. $MAX_PADDED_LEN - 1]];
}

sub greedy_decode {
    my ($model, $src_tok_ids) = @_;
    my $src_batch = [$src_tok_ids];
    my $src_mask  = create_src_mask(src_ids => $src_batch, pad_id => $PAD_ID);
    my $enc_out   = $model->encode(src => $src_batch, src_mask => $src_mask);
    my $dec_in    = [[$SOS_ID, ($PAD_ID) x ($MAX_PADDED_LEN - 1)]];
    my @out;
    for my $t (0 .. $MAX_PADDED_LEN - 2) {
        my $tgt_mask = create_tgt_mask(tgt_ids => $dec_in, pad_id => $PAD_ID);
        my $dec_out  = $model->decode(tgt => $dec_in, encoder_output => $enc_out,
                                      src_mask => $src_mask, tgt_mask => $tgt_mask);
        my $logits   = $model->project(batch => $dec_out);
        my $row      = $logits->[0][$t];
        my $best     = 0;
        for my $v (1 .. $#$row) { $best = $v if $row->[$v] > $row->[$best] }
        push @out, $best;
        last if $best == $EOS_ID;
        $dec_in->[0][$t + 1] = $best if $t + 1 < $MAX_PADDED_LEN;
    }
    return \@out;
}

say "--- Starting Training for $NUM_EPOCHS Epochs ---";

my $train_dataset = ML::DataLoader->new(data => $raw_data, token_to_id => $token_to_id, max_len => $MAX_PADDED_LEN, pad_id => $PAD_ID);

foreach my $epoch (1 .. $NUM_EPOCHS) {
    $train_dataset->init_iter( batch_size => $BATCH_SIZE );
    my $epoch_start_time = [ gettimeofday ];
    my $eff_lr = warmup_lr($epoch, $LEARNING_RATE, $WARMUP_EPOCHS);
    my $avg_epoch_loss = train_one_epoch(model => $transformer_model, dataloader => $train_dataset, loss_fn => "MSE", optimizer => "SGD", device => "CPU", pad_id => $PAD_ID, batch_size => $BATCH_SIZE, learning_rate => $eff_lr);
    my $epoch_duration = sprintf("%.3f", tv_interval( $epoch_start_time , [ gettimeofday ] ));
    say "-" x 50;
    say "End of Epoch $epoch | Time: ${epoch_duration}s | lr: $eff_lr | Avg Loss: $avg_epoch_loss";
    say "-" x 50;
    $transformer_model->save_model( filename => "lr0001_d${d_model}_l${num_layers}_h${num_heads}_${epoch}.json" );
}

# ── greedy-decode accuracy ─────────────────────────────────────────────────
say "=" x 50;
say "--- Greedy Decode Accuracy (200 held-out examples) ---";
my ($correct, $total) = (0, 0);
for my $item (@$test_data) {
    my $src_padded   = tokenize_pad($item->{src});
    my @exp_content  = grep { $_ ne $SOS_TOKEN && $_ ne $EOS_TOKEN && $_ ne $PAD_TOKEN } @{$item->{tgt}};
    my $pred_ids     = greedy_decode($transformer_model, $src_padded);
    my @pred_content = map  { $id_to_token->{$_} }
                       grep { $_ != $EOS_ID && $_ != $SOS_ID && $_ != $PAD_ID } @$pred_ids;
    $correct++ if join(",", @pred_content) eq join(",", @exp_content);
    $total++;
}
my $accuracy = 100.0 * $correct / $total;
my $acc_pass = $accuracy >= 50;
printf "Accuracy: %d/%d = %.1f%%  (pass >= 50%%): %s\n",
    $correct, $total, $accuracy, $acc_pass ? "PASS" : "FAIL";
say "=" x 50;

