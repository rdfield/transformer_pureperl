#!/usr/bin/perl
# test_embedding.pl — correctness tests for ML::Embedding (via ML::InputEmbeddings)
#
# The embedding layer has no differentiable input (token IDs are integers),
# so there is no "input gradient" to verify. Instead we check:
#
#   1. dW accumulation — backward() accumulates into dW[token_id], not dW[seq_pos]
#   2. Repeated tokens — gradient accumulates correctly for repeated token IDs
#   3. sqrt(d_model) scaling — InputEmbeddings scales output by sqrt(d_model);
#      the incoming gradient must be scaled the same way before reaching dW
#   4. optimise() correctness — weight update uses dW[token_id], not dW[seq_pos]
#   5. PAD tokens skipped — optimise() skips PAD positions (no update for PAD id)

use Modern::Perl;
use lib '.';
use ML::InputEmbeddings;
use ML::Embedding;
use ML::LossGradient;
use Storable qw(dclone);
use POSIX qw(sqrt);

my $EPS = 1e-4;
my $TOL = 1e-3;
my ($PASS, $FAIL) = (0, 0);

sub rel_error {
    my ($a, $b) = @_;
    my $denom = abs($a) > abs($b) ? abs($a) : abs($b);
    return 0 if $denom < 1e-12;
    return abs($a - $b) / $denom;
}

sub check {
    my ($label, $analytical, $numerical) = @_;
    my $err = rel_error($analytical, $numerical);
    if ($err < $TOL) {
        printf "  PASS  %-45s  anal=%9.5f  num=%9.5f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $PASS++;
    } else {
        printf "  FAIL  %-45s  anal=%9.5f  num=%9.5f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $FAIL++;
    }
}

sub ok {
    my ($label, $cond) = @_;
    if ($cond) { printf "  PASS  %s\n", $label; $PASS++ }
    else        { printf "  FAIL  %s\n", $label; $FAIL++ }
}

# ── fixed weights ─────────────────────────────────────────────────────────────
# vocab_size=5, d_model=3
my $VOCAB = 5;
my $D     = 3;

my $W = [
    [ 0.1,  0.2,  0.3],   # token 0
    [ 0.4,  0.5,  0.6],   # token 1
    [ 0.7,  0.8,  0.9],   # token 2
    [-0.1, -0.2, -0.3],   # token 3
    [-0.4, -0.5, -0.6],   # token 4
];

# batch of 2 sequences, length 3 each
# token IDs — token 1 appears twice to test gradient accumulation
my $IDS = [
    [0, 1, 2],   # batch 0
    [1, 3, 4],   # batch 1 — token 1 repeated across batches
];

# ── helpers ───────────────────────────────────────────────────────────────────

sub make_emb {
    my $emb = ML::Embedding->new(vocab_size => $VOCAB, embeddings => $D);
    $emb->set_weights(weights => dclone($W));
    return $emb;
}

sub make_input_emb {
    my $emb = ML::InputEmbeddings->new(vocab_size => $VOCAB, embeddings => $D);
    $emb->set_weights(weights => dclone($W));
    return $emb;
}

sub sum_output {
    my ($out) = @_;
    my $s = 0;
    for my $b (@$out) { for my $row (@$b) { $s += $_ for @$row } }
    return $s;
}

sub ones_3d {
    my ($out) = @_;
    my $g = dclone($out);
    for my $b (@$g) { for my $row (@$b) { $_ = 1.0 for @$row } }
    return $g;
}

# ── numerical gradient w.r.t. embedding weights ───────────────────────────────
# loss = sum(all outputs), perturb W[token_id][e] by ±EPS
sub fwd_loss_W {
    my ($W_in, $ids, $scale) = @_;
    my $emb = ML::Embedding->new(vocab_size => $VOCAB, embeddings => $D);
    $emb->set_weights(weights => dclone($W_in));
    my $out = $emb->forward(batch => dclone($ids));
    # apply sqrt(D) scaling as InputEmbeddings does
    if ($scale) {
        for my $b (@$out) { for my $row (@$b) { $_ *= sqrt($D) for @$row } }
    }
    return sum_output($out);
}

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: dW accumulation via ML::Embedding::backward()";
say "=" x 70;
# dW[token_id][e] must accumulate gradients from ALL positions where
# that token appeared across all batches.

{
    my $emb = make_emb();
    my $out = $emb->forward(batch => dclone($IDS));
    my $ones = ones_3d($out);
    $emb->backward(next => ML::LossGradient->new(gradient => $ones));

    # token 0 appears once (batch 0, pos 0): dW[0] = gradient[0][0] = [1,1,1]
    for my $e (0 .. $D-1) {
        my $num = (fwd_loss_W(do { my $Wp=dclone($W); $Wp->[0][$e]+=$EPS; $Wp }, $IDS, 0)
                 - fwd_loss_W(do { my $Wm=dclone($W); $Wm->[0][$e]-=$EPS; $Wm }, $IDS, 0)) / (2*$EPS);
        check("dW[token=0][e=$e]", $emb->{dW}[0][$e], $num);
    }

    # token 1 appears twice (batch 0 pos 1, batch 1 pos 0): dW[1] = [2,2,2]
    for my $e (0 .. $D-1) {
        my $num = (fwd_loss_W(do { my $Wp=dclone($W); $Wp->[1][$e]+=$EPS; $Wp }, $IDS, 0)
                 - fwd_loss_W(do { my $Wm=dclone($W); $Wm->[1][$e]-=$EPS; $Wm }, $IDS, 0)) / (2*$EPS);
        check("dW[token=1][e=$e]  (appears twice, should accumulate)", $emb->{dW}[1][$e], $num);
    }

    # token 4 appears once (batch 1, pos 2)
    for my $e (0 .. $D-1) {
        my $num = (fwd_loss_W(do { my $Wp=dclone($W); $Wp->[4][$e]+=$EPS; $Wp }, $IDS, 0)
                 - fwd_loss_W(do { my $Wm=dclone($W); $Wm->[4][$e]-=$EPS; $Wm }, $IDS, 0)) / (2*$EPS);
        check("dW[token=4][e=$e]", $emb->{dW}[4][$e], $num);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: sqrt(d_model) scaling in ML::InputEmbeddings";
say "=" x 70;
# InputEmbeddings scales output by sqrt(D).  The incoming gradient at the
# embedding table must therefore also be scaled by sqrt(D).

{
    my $ie = make_input_emb();
    my $out = $ie->forward(batch => dclone($IDS));
    my $ones = ones_3d($out);
    $ie->backward(next => ML::LossGradient->new(gradient => $ones));
    my $emb = $ie->{embedding};

    for my $e (0 .. $D-1) {
        my $num = (fwd_loss_W(do { my $Wp=dclone($W); $Wp->[1][$e]+=$EPS; $Wp }, $IDS, 1)
                 - fwd_loss_W(do { my $Wm=dclone($W); $Wm->[1][$e]-=$EPS; $Wm }, $IDS, 1)) / (2*$EPS);
        check("InputEmb dW[token=1][e=$e]  (scaled by sqrt($D))", $emb->{dW}[1][$e], $num);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 3: optimise() applies dW[token_id], not dW[seq_pos]";
say "=" x 70;
# This tests the known bug: optimise() must index dW by token_id, not by i.

{
    my $LR = 0.1;
    my $emb = make_emb();
    my $out = $emb->forward(batch => dclone($IDS));
    $emb->backward(next => ML::LossGradient->new(gradient => ones_3d($out)));

    my @dW_before = map { [@{$emb->{dW}[$_]}] } 0 .. $VOCAB-1;
    my @W_before  = map { [@{$emb->{W}[$_]}]  } 0 .. $VOCAB-1;

    $emb->optimise(learning_rate => $LR);

    # For each token that appeared, verify:
    #   W_new[token_id][e] == W_old[token_id][e] - LR * dW[token_id][e]
    my %seen_tokens;
    for my $b (0 .. $#{$IDS}) {
        for my $s (0 .. $#{$IDS->[$b]}) {
            $seen_tokens{ $IDS->[$b][$s] } = 1;
        }
    }
    for my $tok (sort keys %seen_tokens) {
        for my $e (0 .. $D-1) {
            my $expected = $W_before[$tok][$e] - $LR * $dW_before[$tok][$e];
            check("W[$tok][$e] after SGD: W_old - lr*dW[token_id]",
                $emb->{W}[$tok][$e], $expected);
        }
    }

    # Tokens that did NOT appear should be unchanged
    for my $tok (0 .. $VOCAB-1) {
        next if $seen_tokens{$tok};
        for my $e (0 .. $D-1) {
            ok("W[$tok][$e] unchanged (token not in batch)",
                abs($emb->{W}[$tok][$e] - $W_before[$tok][$e]) < 1e-12);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 4: PAD tokens not updated by optimise()";
say "=" x 70;

{
    my $PAD_ID = 0;
    my $LR = 0.1;
    my $IDS_PAD = [[0, 1, 2], [1, 0, 4]];  # token 0 is PAD, appears twice

    my $emb = ML::Embedding->new(vocab_size => $VOCAB, embeddings => $D);
    $emb->set_weights(weights => dclone($W));
    $emb->{pad_id} = $PAD_ID;

    my $out = $emb->forward(batch => dclone($IDS_PAD));
    $emb->backward(next => ML::LossGradient->new(gradient => ones_3d($out)));

    my @W_before = map { [@{$emb->{W}[$_]}] } 0 .. $VOCAB-1;
    $emb->optimise(learning_rate => $LR);

    # W[PAD_ID] must be unchanged
    for my $e (0 .. $D-1) {
        ok("W[PAD=$PAD_ID][$e] unchanged after optimise()",
            abs($emb->{W}[$PAD_ID][$e] - $W_before[$PAD_ID][$e]) < 1e-12);
    }

    # W[1] should be updated (token 1 appears at non-PAD positions)
    my $token1_changed = 0;
    for my $e (0 .. $D-1) {
        $token1_changed = 1 if abs($emb->{W}[1][$e] - $W_before[1][$e]) > 1e-12;
    }
    ok("W[token=1] updated (non-PAD token)", $token1_changed);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
