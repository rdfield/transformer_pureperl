#!/usr/bin/perl
# test_loss_grad.pl
#
# Numerical gradient check for the cross-entropy loss and its initial gradient.
#
# Method: central finite differences
#   numerical_grad[i][j] = (loss(z + eps*e_ij) - loss(z - eps*e_ij)) / (2*eps)
#
# The code's loss_grad uses grad_scale = batch_size * seq_len (not n_tok) as the
# denominator — this is intentional for Adam stability (see CLAUDE.md).  So:
#   loss_grad[i][j]  ==  analytical_grad[i][j] * (n_tok / grad_scale)
#
# We verify by checking that:
#   loss_grad[i][j] * (grad_scale / n_tok)  ≈  numerical_grad[i][j]

use Modern::Perl;
use lib '.';
use ML::Util qw(softmax);
use POSIX qw(log);

# ── fixed test case ──────────────────────────────────────────────────────────
# 2 positions (B=1, S=2), vocab=6.  Label 0 = PAD (masked).
my $VOCAB_SZ  = 6;
my $PAD_ID    = 0;
my $EPS       = 1e-4;   # finite-difference step
my $TOL       = 1e-5;   # pass/fail tolerance

# Fixed logits — deliberately varied so softmax probs are non-trivial
my $logits = [
    [ 0.5, -1.2,  0.3,  1.1, -0.4,  0.8 ],   # position 0 — PAD label (masked)
    [ 1.3,  0.2, -0.5,  0.9,  0.1, -0.7 ],   # position 1 — label = 3
    [-0.3,  0.6,  1.4, -0.2,  0.8,  0.3 ],   # position 2 — label = 2
    [ 0.7, -0.1,  0.4,  1.2, -0.8,  0.5 ],   # position 3 — label = 1
];
my $labels = [ $PAD_ID, 3, 2, 1 ];

my $N_POS    = scalar @$logits;
my $BATCH_SZ = 1;   # treating flat_out as batch=1 for this check
my $SEQ_LEN  = $N_POS;

# ── loss + gradient (same logic as test_weight_update.pl / train_model.pl) ──
sub compute_loss_and_grad {
    my ($z, $labs) = @_;
    my $sm = softmax($z, 1);   # row-wise softmax
    my $n_tok      = scalar grep { $_ != $PAD_ID } @$labs;
    my $loss_scale = $n_tok || 1;
    my $grad_scale = $BATCH_SZ * $SEQ_LEN;

    my $loss = 0;
    my $grad = [];
    for my $i (0 .. $N_POS - 1) {
        if ($labs->[$i] == $PAD_ID) {
            $grad->[$i][$_] = 0 for 0 .. $VOCAB_SZ - 1;
            next;
        }
        for my $j (0 .. $VOCAB_SZ - 1) {
            if ($j == $labs->[$i]) {
                $loss += -log($sm->[$i][$j]) / $loss_scale;
                $grad->[$i][$j] = ($sm->[$i][$j] - 1) / $grad_scale;
            } else {
                $grad->[$i][$j] = $sm->[$i][$j] / $grad_scale;
            }
        }
    }
    return ($loss, $grad, $n_tok, $grad_scale);
}

# ── numerical gradient ───────────────────────────────────────────────────────
sub perturb_loss {
    my ($z, $labs, $pi, $pj, $delta) = @_;
    # deep copy and perturb one element
    my $z2 = [ map { [@$_] } @$z ];
    $z2->[$pi][$pj] += $delta;
    my ($loss) = compute_loss_and_grad($z2, $labs);
    return $loss;
}

# ── run check ────────────────────────────────────────────────────────────────
my ($loss, $agrad, $n_tok, $grad_scale) = compute_loss_and_grad($logits, $labels);
printf "Loss = %.6f  (n_tok=%d, grad_scale=%d)\n\n", $loss, $n_tok, $grad_scale;

my $pass = 0; my $fail = 0;
for my $i (0 .. $N_POS - 1) {
    for my $j (0 .. $VOCAB_SZ - 1) {
        my $lp = perturb_loss($logits, $labels, $i, $j, +$EPS);
        my $lm = perturb_loss($logits, $labels, $i, $j, -$EPS);
        my $num_grad  = ($lp - $lm) / (2 * $EPS);                   # true dL/dz
        my $ana_grad  = $agrad->[$i][$j] * $grad_scale / $n_tok;    # rescale to true dL/dz
        my $err = abs($num_grad - $ana_grad);
        my $rel = $num_grad != 0 ? $err / abs($num_grad) : $err;
        if ($rel > $TOL && $err > $TOL) {
            printf "FAIL pos=%d vocab=%d  num=%.8f  ana=%.8f  err=%.2e\n",
                   $i, $j, $num_grad, $ana_grad, $err;
            $fail++;
        } else {
            $pass++;
        }
    }
}
printf "\n%d/%d pass  (tol=%.0e)\n", $pass, $pass+$fail, $TOL;

# ── also print softmax probs vs labels for a sanity read ────────────────────
say "\nSoftmax probabilities vs labels:";
my $sm = softmax($logits, 1);
for my $i (0 .. $N_POS - 1) {
    my $lbl = $labels->[$i];
    my $p_correct = $lbl == $PAD_ID ? "PAD(masked)" : sprintf("%.4f", $sm->[$i][$lbl]);
    printf "  pos %d  label=%-3s  p(correct)=%s\n", $i, $lbl, $p_correct;
}
