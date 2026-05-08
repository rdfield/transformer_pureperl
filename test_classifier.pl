#!/usr/bin/perl
# test_classifier.pl — toy classifier convergence test.
# Encoder-only architecture with [CLS] token and Linear classification head.
# Task: classify a sequence by which of {A,B,C} appears most frequently.
# (D and E are filler/noise tokens.)

use Modern::Perl;
use lib '.';
use ML::InputEmbeddings;
use ML::PositionalEmbeddings;
use ML::Encoder;
use ML::Linear;
use ML::LossGradient;
use ML::Util qw(create_src_mask softmax global_clip_grad_norm print_2d_array);
use Time::HiRes qw(tv_interval gettimeofday);
use Data::Dumper;

# ── vocabulary ────────────────────────────────────────────────────────────────
my $PAD = '[PAD]';
my $CLS = '[CLS]';
my @CONTENT = ('A', 'B', 'C', 'D', 'E');
my $VOCAB   = [$PAD, $CLS, @CONTENT];
my (%tok2id, %id2tok);
for my $i (0 .. $#$VOCAB) {
    $tok2id{ $VOCAB->[$i] } = $i;
    $id2tok{$i}             = $VOCAB->[$i];
}
my $VOCAB_SIZE  = scalar @$VOCAB;
my $PAD_ID      = $tok2id{$PAD};
my $CLS_ID      = $tok2id{$CLS};
my $NUM_CLASSES = 3;   # 0 = A-majority, 1 = B-majority, 2 = C-majority

# ── hyper-parameters ──────────────────────────────────────────────────────────
my $D_MODEL     = 12;
my $NUM_LAYERS  = 2;
my $NUM_HEADS   = 4;
my $D_FF        = 48;
my $DROPOUT     = "0.01";
my $MIN_LEN     = 4;
my $MAX_LEN     = 6;
my $MAX_PAD_LEN = $MAX_LEN + 1;   # CLS + up-to-MAX_LEN content tokens
my $NUM_TRAIN   = 240;
my $NUM_TEST    = 40;
my $BATCH_SIZE  = 8;
my $NUM_EPOCHS  = 50;
my $LR          = 0.001;
my $WARMUP_EPOCHS = 5;
my $MIN_LR      = $LR * 0.1;

my $LOSS_PASS_RATIO   = 0.5;
my $ACCURACY_PASS_PCT = 80;

# ── data generation ───────────────────────────────────────────────────────────
# Pick a winner W in {A,B,C}, then build a sequence with W appearing more
# often than the other two scoring tokens. D/E are filler.
sub gen_data {
    my ($n, $seed) = @_;
    srand($seed // 1);
    my @data;
    my @winners = ('A', 'B', 'C');
    for (1 .. $n) {
        my $win    = $winners[ int rand 3 ];
        my $class  = $win eq 'A' ? 0 : $win eq 'B' ? 1 : 2;
        my $len    = $MIN_LEN + int(rand($MAX_LEN - $MIN_LEN + 1));
        my @others = grep { $_ ne $win } @CONTENT;   # 4 fillers/competitors

        # Place enough winners to guarantee strict majority among A/B/C
        my $n_win  = int($len / 2) + 1;
        my @seq    = ($win) x $n_win;
        push @seq, $others[ int rand @others ] for 1 .. ($len - $n_win);

        # Shuffle in place
        for (my $i = $#seq; $i > 0; $i--) {
            my $j = int rand($i + 1);
            @seq[$i, $j] = @seq[$j, $i];
        }
        push @data, { seq => [$CLS, @seq], class => $class };
    }
    return \@data;
}

sub tokenize_pad {
    my ($tokens) = @_;
    my @ids = map { $tok2id{$_} } @$tokens;
    while (@ids < $MAX_PAD_LEN) { push @ids, $PAD_ID }
    return [@ids[0 .. $MAX_PAD_LEN - 1]];
}

# ── model construction ────────────────────────────────────────────────────────
sub build_model {
    srand(1);
    my $emb = ML::InputEmbeddings->new(embeddings => $D_MODEL, vocab_size => $VOCAB_SIZE);
    my $pos = ML::PositionalEmbeddings->new(embeddings => $D_MODEL, max_seq_length => $MAX_PAD_LEN, dropout => $DROPOUT);
    my $enc = ML::Encoder->new(embeddings => $D_MODEL, layers => $NUM_LAYERS,
                               max_seq_length => $MAX_PAD_LEN, dropout => $DROPOUT,
                               num_heads => $NUM_HEADS, d_ff => $D_FF);
    my $cls = ML::Linear->new(insize => $D_MODEL, outsize => $NUM_CLASSES);
    $cls->set_weights_and_biases();
    return { emb => $emb, pos => $pos, enc => $enc, cls => $cls };
}

# ── training ──────────────────────────────────────────────────────────────────
sub cosine_lr {
    my ($epoch, $lr, $warmup, $total, $min_lr) = @_;
    $min_lr //= $lr * 0.1;
    if ($epoch <= $warmup) {
       return $lr * $epoch / $warmup ;
    }
    my $pi       = 4 * atan2(1, 1);
    my $progress = ($epoch - $warmup) / ($total - $warmup);
    return $min_lr + 0.5 * ($lr - $min_lr) * (1 + cos($pi * $progress));
}

sub train_batch {
    my ($model, $inputs, $labels, $lr) = @_;
    my $bs       = scalar @$inputs;
    my $src_mask = create_src_mask(src_ids => $inputs, pad_id => $PAD_ID);

    my $emb_out = $model->{emb}->forward(batch => $inputs);
    my $pos_out = $model->{pos}->forward(batch => $emb_out);
    my $enc_out = $model->{enc}->forward(batch => $pos_out, mask => $src_mask);

    # Take the CLS position (index 0) from each batch element → [batch, d_model]
    my $cls_vecs = [];
    for my $b (0 .. $bs - 1) {
        $cls_vecs->[$b] = [ @{ $enc_out->[$b][0] } ];
    }
print_2d_array("cls vecs", $cls_vecs);

    # Classification head: [batch, d_model] → [batch, num_classes]
    my $logits = $model->{cls}->forward(batch => $cls_vecs);

print_2d_array("logits", $logits);
    # Cross-entropy loss + gradient
    my $sm        = softmax($logits, 1);   # row-wise
print_2d_array("softmax", $sm);
    my $loss      = 0;
    my $loss_grad = [];
    for my $b (0 .. $bs - 1) {
        my $true = $labels->[$b];
        $loss   += -log($sm->[$b][$true] + 1e-12);
        for my $c (0 .. $NUM_CLASSES - 1) {
            $loss_grad->[$b][$c] = ($sm->[$b][$c] - ($c == $true ? 1 : 0)) / $bs;
        }
    }
    $loss /= $bs;
print_2d_array("loss_grad", $loss_grad);
say "loss = $loss";

    # Backward
    my $next = ML::LossGradient->new(gradient => $loss_grad);
    $model->{cls}->backward(next => $next);
print_2d_array("cls bias grad", $model->{cls}{bias_grad});
print_2d_array("cls weights grad", $model->{cls}{weights_grad});
print_2d_array("cls gradient", $model->{cls}->gradient());
    # cls.gradient() is [batch, d_model]; expand to [batch, seq, d_model]
    # with non-CLS positions zeroed.
    my $cls_grad = $model->{cls}->gradient();
    my $expanded = [];
    for my $b (0 .. $bs - 1) {
        for my $s (0 .. $MAX_PAD_LEN - 1) {
            for my $d (0 .. $D_MODEL - 1) {
                $expanded->[$b][$s][$d] = ($s == 0) ? $cls_grad->[$b][$d] : 0;
            }
        }
print_2d_array("expanded $b", $expanded->[$b]);
    }

    my $enc_src = ML::LossGradient->new(gradient => $expanded);
    $model->{enc}->backward(next => $enc_src);
    my $enc_grad = $model->{enc}->gradient();
    for my $b (0 .. $bs - 1) {
       print_2d_array("enc_grad $b", $enc_grad->[$b]);
    }
    $model->{pos}->backward(next => $model->{enc});
    $model->{emb}->backward(next => $model->{pos});

    # Global gradient clipping (same threshold as test_convergence.pl)
    my @tensors;
    push @tensors, @{ $model->{cls}->get_grad_tensors() };
    push @tensors, @{ $model->{enc}->get_grad_tensors() };
    push @tensors, @{ $model->{emb}->get_grad_tensors() };
    my $gnorm = global_clip_grad_norm(\@tensors, 1.0);

    # Optimise
    $model->{cls}->optimise(learning_rate => $lr);
    $model->{enc}->optimise(learning_rate => $lr);
    $model->{pos}->optimise(learning_rate => $lr);
    $model->{emb}->optimise(learning_rate => $lr);

    return ($loss, $gnorm);
}

sub train_epoch {
    my ($model, $data, $lr) = @_;
    my @shuffled = sort { rand() <=> rand() } @$data;
    my ($total_loss, $total_gnorm, $n_batches) = (0, 0, 0);
    for (my $i = 0; $i + $BATCH_SIZE <= scalar @shuffled; $i += $BATCH_SIZE) {
        my @batch  = @shuffled[ $i .. $i + $BATCH_SIZE - 1 ];
        my $inputs = [ map { tokenize_pad($_->{seq}) } @batch ];
        my $labels = [ map { $_->{class} } @batch ];
        my ($loss, $gn) = train_batch($model, $inputs, $labels, $lr);
        $total_loss  += $loss;
        $total_gnorm += $gn;
        $n_batches++;
    }
    return ($total_loss / $n_batches, $total_gnorm / $n_batches);
}

sub classify {
    my ($model, $seq_tokens) = @_;
    my $padded   = tokenize_pad($seq_tokens);
    my $batch    = [$padded];
    my $src_mask = create_src_mask(src_ids => $batch, pad_id => $PAD_ID);
    my $emb_out  = $model->{emb}->forward(batch => $batch);
    my $pos_out  = $model->{pos}->forward(batch => $emb_out);
    my $enc_out  = $model->{enc}->forward(batch => $pos_out, mask => $src_mask);
    my $cls_vec  = [ [ @{ $enc_out->[0][0] } ] ];   # [1, d_model]
    my $logits   = $model->{cls}->forward(batch => $cls_vec);
    my $best     = 0;
    for my $c (1 .. $NUM_CLASSES - 1) {
        $best = $c if $logits->[0][$c] > $logits->[0][$best];
    }
    return $best;
}

# ── run ───────────────────────────────────────────────────────────────────────
say "Vocab ($VOCAB_SIZE): " . join(", ", @$VOCAB);
say "Classes: $NUM_CLASSES (0=A-majority, 1=B-majority, 2=C-majority; D/E are filler)";
say "Model: d=$D_MODEL  layers=$NUM_LAYERS  heads=$NUM_HEADS  d_ff=$D_FF";
say "Data: $NUM_TRAIN train / $NUM_TEST test  seq=$MIN_LEN..$MAX_LEN  padded=$MAX_PAD_LEN";
say "-" x 60;

my $train_data = gen_data($NUM_TRAIN, 1);
my $test_data  = gen_data($NUM_TEST,  99);
my $model      = build_model();
my ($first_loss, $last_loss);

for my $epoch (1 .. $NUM_EPOCHS) {
    my $t0      = [gettimeofday];
    my $eff_lr  = cosine_lr($epoch, $LR, $WARMUP_EPOCHS, $NUM_EPOCHS, $MIN_LR);
    my ($avg_loss, $avg_gnorm) = train_epoch($model, $train_data, $eff_lr);
    my $elapsed = sprintf "%.1f", tv_interval($t0, [gettimeofday]);
    $first_loss //= $avg_loss;
    $last_loss    = $avg_loss;
    printf "Epoch %3d | loss %.4f | lr %.5f | gnorm %.2f | %.1fs\n",
        $epoch, $avg_loss, $eff_lr, $avg_gnorm, $elapsed;
}

# ── convergence check ─────────────────────────────────────────────────────────
say "-" x 60;
my $ratio     = $last_loss / $first_loss;
my $loss_pass = $ratio < $LOSS_PASS_RATIO;
printf "Loss: epoch-1 = %.4f  final = %.4f  ratio = %.3f  (pass < %.2f): %s\n",
    $first_loss, $last_loss, $ratio, $LOSS_PASS_RATIO, $loss_pass ? "PASS" : "FAIL";

# ── accuracy ──────────────────────────────────────────────────────────────────
my ($correct, $total) = (0, 0);
for my $item (@$test_data) {
    my $pred = classify($model, $item->{seq});
    $correct++ if $pred == $item->{class};
    $total++;
}
my $accuracy = 100.0 * $correct / $total;
my $acc_pass = $accuracy >= $ACCURACY_PASS_PCT;
printf "Accuracy: %d/%d = %.1f%%  (pass >= %d%%): %s\n",
    $correct, $total, $accuracy, $ACCURACY_PASS_PCT, $acc_pass ? "PASS" : "FAIL";

say "=" x 60;
say "OVERALL: " . (($loss_pass && $acc_pass) ? "PASS" : "FAIL");
