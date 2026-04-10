#!/usr/bin/perl
# test_linear.pl — gradient check for ML::Linear
#
# Uses central finite differences as the ground truth.
# For each parameter θ: numerical_grad = (L(θ+ε) - L(θ-ε)) / 2ε
# Compares against the analytical gradient from backward().
#
# Loss: L = sum of all output elements  →  dL/doutput = ones everywhere.
# This is the simplest possible loss and gives clean, checkable values.
#
# Tests:
#   1. 2D input  [seq_len × insize]       — non-batched forward path
#   2. 3D input  [batch × seq_len × insize] — batched path (used by projection layer)
#   Each test checks: weights_grad, bias_grad, and input gradient.

use Modern::Perl;
use lib '.';
use ML::Linear;
use ML::LossGradient;
use Storable qw(dclone);
use POSIX qw(sqrt);

my $EPS  = 1e-4;   # finite difference step size
my $TOL  = 1e-3;   # max acceptable relative error
my ($PASS, $FAIL) = (0, 0);

# ── helpers ───────────────────────────────────────────────────────────────────
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
        printf "  PASS  %-35s  anal=%10.6f  num=%10.6f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $PASS++;
    } else {
        printf "  FAIL  %-35s  anal=%10.6f  num=%10.6f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $FAIL++;
    }
}

sub sum_output {
    my ($out) = @_;
    my $s = 0;
    if (ref($out->[0][0]) eq 'ARRAY') {   # 3D
        for my $b (@$out) { for my $row (@$b) { $s += $_ for @$row } }
    } else {                                # 2D
        for my $row (@$out) { $s += $_ for @$row }
    }
    return $s;
}

sub make_linear {
    my ($insize, $outsize, $W, $b) = @_;
    my $lin = ML::Linear->new(insize => $insize, outsize => $outsize);
    $lin->set_weights_and_biases(weights => dclone($W), biases => dclone($b));
    return $lin;
}

sub fwd_loss {
    my ($insize, $outsize, $W, $b, $input) = @_;
    my $lin = make_linear($insize, $outsize, $W, $b);
    return sum_output($lin->forward(batch => dclone($input)));
}

sub ones_like {
    my ($ref) = @_;
    my $g = dclone($ref);
    if (ref($g->[0][0]) eq 'ARRAY') {
        for my $b (@$g) { for my $row (@$b) { $_ = 1.0 for @$row } }
    } else {
        for my $row (@$g) { $_ = 1.0 for @$row }
    }
    return $g;
}

# ── fixed weights and biases ──────────────────────────────────────────────────
my $INSIZE  = 3;
my $OUTSIZE = 4;

my $W = [
    [ 0.4, -0.3,  0.7,  0.2],
    [-0.2,  0.8, -0.5,  0.1],
    [ 0.9,  0.1,  0.3, -0.6],
];
my $BIAS = [   # [outsize × 1]
    [ 0.1], [-0.2], [ 0.3], [-0.1],
];

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: 2D input [seq_len=${\($INSIZE-1)} x insize=$INSIZE]";
say "=" x 70;

{
    my $input = [
        [ 0.5, -1.2,  0.8],
        [ 1.1,  0.3, -0.6],
    ];
    my $SEQ = scalar @$input;

    my $lin = make_linear($INSIZE, $OUTSIZE, $W, $BIAS);
    my $out = $lin->forward(batch => dclone($input));
    $lin->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    say "\nweights_grad [insize x outsize]:";
    for my $i (0 .. $INSIZE-1) {
        for my $j (0 .. $OUTSIZE-1) {
            my ($Wp, $Wm) = (dclone($W), dclone($W));
            $Wp->[$i][$j] += $EPS;  $Wm->[$i][$j] -= $EPS;
            my $num = (fwd_loss($INSIZE,$OUTSIZE,$Wp,$BIAS,$input)
                     - fwd_loss($INSIZE,$OUTSIZE,$Wm,$BIAS,$input)) / (2*$EPS);
            check("W[$i][$j]", $lin->{weights_grad}[$i][$j], $num);
        }
    }

    say "\nbias_grad [outsize x 1]:";
    for my $j (0 .. $OUTSIZE-1) {
        my ($bp, $bm) = (dclone($BIAS), dclone($BIAS));
        $bp->[$j][0] += $EPS;  $bm->[$j][0] -= $EPS;
        my $num = (fwd_loss($INSIZE,$OUTSIZE,$W,$bp,$input)
                 - fwd_loss($INSIZE,$OUTSIZE,$W,$bm,$input)) / (2*$EPS);
        check("bias[$j]", $lin->{bias_grad}[$j][0], $num);
    }

    say "\ninput gradient [seq_len x insize]:";
    for my $s (0 .. $SEQ-1) {
        for my $i (0 .. $INSIZE-1) {
            my ($ip, $im) = (dclone($input), dclone($input));
            $ip->[$s][$i] += $EPS;  $im->[$s][$i] -= $EPS;
            my $num = (fwd_loss($INSIZE,$OUTSIZE,$W,$BIAS,$ip)
                     - fwd_loss($INSIZE,$OUTSIZE,$W,$BIAS,$im)) / (2*$EPS);
            my $grad = $lin->gradient();
            check("input[$s][$i]", $grad->[$s][$i], $num);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: 3D input [batch=2 x seq_len=3 x insize=$INSIZE]  (projection layer path)";
say "=" x 70;

{
    my $input3d = [
        [[ 0.5, -1.2,  0.8], [ 1.1,  0.3, -0.6], [-0.4,  0.7,  0.2]],
        [[-0.3,  0.9,  1.4], [ 0.6, -0.8,  0.1], [ 0.2,  0.5, -1.1]],
    ];
    my $BATCH = scalar @$input3d;
    my $SEQ   = scalar @{$input3d->[0]};

    my $lin = make_linear($INSIZE, $OUTSIZE, $W, $BIAS);
    my $out = $lin->forward(batch => dclone($input3d));
    $lin->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    say "\nweights_grad [insize x outsize]:";
    for my $i (0 .. $INSIZE-1) {
        for my $j (0 .. $OUTSIZE-1) {
            my ($Wp, $Wm) = (dclone($W), dclone($W));
            $Wp->[$i][$j] += $EPS;  $Wm->[$i][$j] -= $EPS;
            my $num = (fwd_loss($INSIZE,$OUTSIZE,$Wp,$BIAS,$input3d)
                     - fwd_loss($INSIZE,$OUTSIZE,$Wm,$BIAS,$input3d)) / (2*$EPS);
            check("W[$i][$j]", $lin->{weights_grad}[$i][$j], $num);
        }
    }

    say "\nbias_grad [outsize x 1]:";
    for my $j (0 .. $OUTSIZE-1) {
        my ($bp, $bm) = (dclone($BIAS), dclone($BIAS));
        $bp->[$j][0] += $EPS;  $bm->[$j][0] -= $EPS;
        my $num = (fwd_loss($INSIZE,$OUTSIZE,$W,$bp,$input3d)
                 - fwd_loss($INSIZE,$OUTSIZE,$W,$bm,$input3d)) / (2*$EPS);
        check("bias[$j]", $lin->{bias_grad}[$j][0], $num);
    }

    say "\ninput gradient [batch x seq_len x insize]:";
    for my $b (0 .. $BATCH-1) {
        for my $s (0 .. $SEQ-1) {
            for my $i (0 .. $INSIZE-1) {
                my ($ip, $im) = (dclone($input3d), dclone($input3d));
                $ip->[$b][$s][$i] += $EPS;  $im->[$b][$s][$i] -= $EPS;
                my $num = (fwd_loss($INSIZE,$OUTSIZE,$W,$BIAS,$ip)
                         - fwd_loss($INSIZE,$OUTSIZE,$W,$BIAS,$im)) / (2*$EPS);
                my $grad = $lin->gradient();
                check("input[$b][$s][$i]", $grad->[$b][$s][$i], $num);
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
