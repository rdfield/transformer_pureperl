#!/usr/bin/perl
# test_residual.pl — gradient correctness for ML::ResidualConnection
#
# Architecture: output = LayerNorm( sublayer(input) + input )
# Sublayer: PositionwiseFeedForward (verified correct by test_ffn.pl)
#
# NOTE: dL/dout = all-ones is useless here because the LN input gradient
# sums to exactly zero in that case (LN is translation-invariant when
# gamma=1).  We use a non-constant upstream gradient instead, expressed as
# a weighted sum loss: L = sum_{b,s,e} output[b][s][e] * G[b][s][e].
#
# Tests:
#   1. input gradient  — exercises the skip connection path
#   2. LN gamma/beta gradients
#   3. PFF L1 weight gradients (non-zero only with non-constant upstream)

use Modern::Perl;
use lib '.';
use ML::ResidualConnection;
use ML::PositionwiseFeedForward;
use ML::LossGradient;
use Storable qw(dclone);

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
    my ($label, $anal, $num) = @_;
    my $err = rel_error($anal, $num);
    if ($err < $TOL || abs(($anal // 0) - ($num // 0)) < $EPS) {
        printf "  PASS  %-44s  anal=%9.5f  num=%9.5f  err=%.2e\n", $label, $anal, $num, $err;
        $PASS++;
    } else {
        printf "  FAIL  %-44s  anal=%9.5f  num=%9.5f  err=%.2e\n", $label, $anal, $num, $err;
        $FAIL++;
    }
}

sub weighted_sum {
    my ($out, $w) = @_;
    my $s = 0;
    for my $b (0 .. $#$out) {
        for my $r (0 .. $#{$out->[$b]}) {
            for my $e (0 .. $#{$out->[$b][$r]}) {
                $s += $out->[$b][$r][$e] * $w->[$b][$r][$e];
            }
        }
    }
    return $s;
}

my $D     = 4;
my $D_FF  = 6;
my $BATCH = 2;
my $SEQ   = 3;

my $W1 = [
    [ 0.3, -0.2,  0.5,  0.1, -0.4,  0.2],
    [-0.1,  0.4, -0.3,  0.6,  0.2, -0.5],
    [ 0.5, -0.1,  0.2, -0.4,  0.3,  0.1],
    [-0.3,  0.6,  0.1, -0.2,  0.5, -0.4],
];
my $B1 = [[0.1],[-0.1],[0.2],[-0.2],[0.1],[-0.1]];
my $W2 = [
    [ 0.2, -0.3,  0.4, -0.1],
    [-0.4,  0.1,  0.3,  0.2],
    [ 0.1,  0.5, -0.2,  0.3],
    [-0.2,  0.3,  0.1, -0.5],
    [ 0.4, -0.1,  0.2,  0.3],
    [ 0.3,  0.2, -0.4,  0.1],
];
my $B2 = [[0.1],[-0.2],[0.1],[-0.1]];

my $INPUT = [
    [[ 0.5, -1.2,  0.8,  0.3],
     [ 1.1,  0.3, -0.6,  0.9],
     [-0.4,  0.7,  0.2, -0.8]],
    [[-0.3,  0.9,  1.4, -0.5],
     [ 0.6, -0.8,  0.1,  1.2],
     [ 0.2,  0.5, -1.1,  0.4]],
];

# Non-constant upstream gradient — avoids the all-ones cancellation in LN.
my $GRAD = [
    [[ 0.3, -0.1,  0.5, -0.2],
     [-0.4,  0.2,  0.1,  0.3],
     [ 0.2, -0.3, -0.1,  0.4]],
    [[ 0.1,  0.4, -0.3,  0.2],
     [-0.2, -0.1,  0.4,  0.3],
     [ 0.5,  0.1, -0.2, -0.4]],
];

sub make_pff {
    my ($w1,$b1,$w2,$b2) = @_;
    return ML::PositionwiseFeedForward->new(
        embeddings => $D, d_ff => $D_FF, dropout => undef,
        l1 => { weights => dclone($w1), biases => dclone($b1) },
        l2 => { weights => dclone($w2), biases => dclone($b2) },
    );
}

sub make_rc {
    return ML::ResidualConnection->new(
        embeddings => $D, max_seq_length => $SEQ, dropout => undef,
    );
}

# Forward pass + weighted-sum loss
sub fwd_loss {
    my ($w1,$b1,$w2,$b2,$input) = @_;
    my $pff = make_pff($w1,$b1,$w2,$b2);
    my $rc  = make_rc();
    my $out = $rc->forward(batch => dclone($input), sublayer => $pff);
    return weighted_sum($out, $GRAD);
}

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: ResidualConnection — input gradient [batch=$BATCH × seq=$SEQ × d=$D]";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $rc  = make_rc();
    my $out = $rc->forward(batch => dclone($INPUT), sublayer => $pff);
    $rc->backward(next => ML::LossGradient->new(gradient => dclone($GRAD)));

    for my $b (0 .. $BATCH-1) {
        for my $s (0 .. $SEQ-1) {
            for my $e (0 .. $D-1) {
                my ($ip,$im) = (dclone($INPUT),dclone($INPUT));
                $ip->[$b][$s][$e] += $EPS;  $im->[$b][$s][$e] -= $EPS;
                my $num = (fwd_loss($W1,$B1,$W2,$B2,$ip) - fwd_loss($W1,$B1,$W2,$B2,$im)) / (2*$EPS);
                check("dinput[$b][$s][$e]", $rc->{gradient}[$b][$s][$e], $num);
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: ResidualConnection — LN gamma/beta gradients";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $rc  = make_rc();
    my $out = $rc->forward(batch => dclone($INPUT), sublayer => $pff);
    $rc->backward(next => ML::LossGradient->new(gradient => dclone($GRAD)));

    for my $e (0 .. $D-1) {
        my $num_g = do {
            my $rc2 = make_rc(); $rc2->{ln_layer}{gamma}[$e] += $EPS;
            my $pff2 = make_pff($W1,$B1,$W2,$B2);
            my $op = weighted_sum($rc2->forward(batch => dclone($INPUT), sublayer => $pff2), $GRAD);
            my $rc3 = make_rc(); $rc3->{ln_layer}{gamma}[$e] -= $EPS;
            my $pff3 = make_pff($W1,$B1,$W2,$B2);
            my $om = weighted_sum($rc3->forward(batch => dclone($INPUT), sublayer => $pff3), $GRAD);
            ($op - $om) / (2*$EPS);
        };
        check("LN_gamma[$e]", $rc->{ln_layer}{dgamma}[$e], $num_g);

        my $num_b = do {
            my $rc2 = make_rc(); $rc2->{ln_layer}{beta}[$e] += $EPS;
            my $pff2 = make_pff($W1,$B1,$W2,$B2);
            my $op = weighted_sum($rc2->forward(batch => dclone($INPUT), sublayer => $pff2), $GRAD);
            my $rc3 = make_rc(); $rc3->{ln_layer}{beta}[$e] -= $EPS;
            my $pff3 = make_pff($W1,$B1,$W2,$B2);
            my $om = weighted_sum($rc3->forward(batch => dclone($INPUT), sublayer => $pff3), $GRAD);
            ($op - $om) / (2*$EPS);
        };
        check("LN_beta[$e]",  $rc->{ln_layer}{dbeta}[$e],  $num_b);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 3: ResidualConnection — PFF L1 weights gradient [d=$D × d_ff=$D_FF]";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $rc  = make_rc();
    my $out = $rc->forward(batch => dclone($INPUT), sublayer => $pff);
    $rc->backward(next => ML::LossGradient->new(gradient => dclone($GRAD)));

    for my $i (0 .. $D-1) {
        for my $j (0 .. $D_FF-1) {
            my ($wp,$wm) = (dclone($W1),dclone($W1));
            $wp->[$i][$j] += $EPS;  $wm->[$i][$j] -= $EPS;
            my $num = (fwd_loss($wp,$B1,$W2,$B2,$INPUT) - fwd_loss($wm,$B1,$W2,$B2,$INPUT)) / (2*$EPS);
            check("L1_W[$i][$j]", $pff->{l1}{weights_grad}[$i][$j], $num);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
