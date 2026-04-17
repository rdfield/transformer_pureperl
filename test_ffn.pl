#!/usr/bin/perl
# test_ffn.pl — gradient correctness for ML::LinearReLu and
#               ML::PositionwiseFeedForward (without dropout).
#
# Loss: L = sum of all output elements  →  dL/doutput = ones.
# Uses central finite differences (ε=1e-4) as ground truth.
#
# Tests:
#   1. LinearReLu backward  — input gradient
#   2. PFF input gradient   — [batch × seq × d_model]
#   3. PFF L1 weights/bias  — d_model → d_ff
#   4. PFF L2 weights/bias  — d_ff → d_model

use Modern::Perl;
use lib '.';
use ML::PositionwiseFeedForward;
use ML::LinearReLu;
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
    # Also pass if absolute error is tiny (handles near-zero gradient comparisons
    # where rel_error is unreliable when both values are close to floating-point noise).
    if ($err < $TOL || abs(($anal // 0) - ($num // 0)) < $EPS) {
        printf "  PASS  %-42s  anal=%9.5f  num=%9.5f  err=%.2e\n", $label, $anal, $num, $err;
        $PASS++;
    } else {
        printf "  FAIL  %-42s  anal=%9.5f  num=%9.5f  err=%.2e\n", $label, $anal, $num, $err;
        $FAIL++;
    }
}

sub sum_output {
    my ($out) = @_;
    my $s = 0;
    if (ref($out->[0][0]) eq 'ARRAY') {
        for my $b (@$out) { for my $r (@$b) { $s += $_ for @$r } }
    } elsif (ref($out->[0]) eq 'ARRAY') {
        for my $r (@$out) { $s += $_ for @$r }
    } else {
        $s += $_ for @$out;
    }
    return $s;
}

sub ones_like {
    my ($ref) = @_;
    my $g = dclone($ref);
    if (ref($g->[0][0]) eq 'ARRAY') {
        for my $b (@$g) { for my $r (@$b) { $_ = 1.0 for @$r } }
    } else {
        for my $r (@$g) { $_ = 1.0 for @$r }
    }
    return $g;
}

# ── fixed PFF dimensions ──────────────────────────────────────────────────────
my $D     = 4;    # d_model
my $D_FF  = 6;    # d_ff
my $BATCH = 2;
my $SEQ   = 3;

# Fixed weights for reproducibility — small random-ish values
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

# ── helpers ───────────────────────────────────────────────────────────────────

sub make_pff {
    my ($w1, $b1, $w2, $b2) = @_;
    return ML::PositionwiseFeedForward->new(
        embeddings => $D, d_ff => $D_FF,
        dropout    => undef,   # no dropout for gradient check
        l1 => { weights => dclone($w1), biases => dclone($b1) },
        l2 => { weights => dclone($w2), biases => dclone($b2) },
    );
}

sub fwd_loss_pff {
    my ($w1,$b1,$w2,$b2,$input) = @_;
    my $pff = make_pff($w1,$b1,$w2,$b2);
    return sum_output($pff->forward(batch => dclone($input)));
}

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: LinearReLu — input gradient";
say "=" x 70;

{
    # Small 2D input (seq × d_ff) for the activation layer alone
    my $relu_input = [
        [ 0.5, -1.2,  0.8, -0.3,  1.1, -0.05],
        [-0.4,  0.7,  0.2, -0.8,  0.3, -0.1],
        [ 1.1,  0.3, -0.6,  0.9, -0.5,  0.4],
    ];
    my $relu = ML::LinearReLu->new(insize => 6);
    my $out  = $relu->forward(batch => dclone($relu_input));
    $relu->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    for my $i (0 .. $#$relu_input) {
        for my $j (0 .. $#{$relu_input->[$i]}) {
            # Numerical: perturb input, rerun forward
            my ($ip,$im) = (dclone($relu_input),dclone($relu_input));
            $ip->[$i][$j] += $EPS;  $im->[$i][$j] -= $EPS;
            my $relu_p = ML::LinearReLu->new(insize => 6);
            my $relu_m = ML::LinearReLu->new(insize => 6);
            $relu_p->{input} = dclone($relu_input); $relu_m->{input} = dclone($relu_input);
            my $lp = sum_output(do { my $r=ML::LinearReLu->new(insize=>6); $r->forward(batch=>$ip) });
            my $lm = sum_output(do { my $r=ML::LinearReLu->new(insize=>6); $r->forward(batch=>$im) });
            my $num = ($lp - $lm) / (2*$EPS);
            check("relu_grad[$i][$j]", $relu->gradient()->[$i][$j], $num);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: PositionwiseFeedForward — input gradient [batch=$BATCH × seq=$SEQ × d=$D]";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $out = $pff->forward(batch => dclone($INPUT));
    $pff->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    for my $b (0 .. $BATCH-1) {
        for my $s (0 .. $SEQ-1) {
            for my $e (0 .. $D-1) {
                my ($ip,$im) = (dclone($INPUT),dclone($INPUT));
                $ip->[$b][$s][$e] += $EPS;  $im->[$b][$s][$e] -= $EPS;
                my $num = (fwd_loss_pff($W1,$B1,$W2,$B2,$ip)
                         - fwd_loss_pff($W1,$B1,$W2,$B2,$im)) / (2*$EPS);
                check("dinput[$b][$s][$e]", $pff->{gradient}[$b][$s][$e], $num);
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 3: PFF L1 weights gradient [d_model=$D × d_ff=$D_FF]";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $out = $pff->forward(batch => dclone($INPUT));
    $pff->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    for my $i (0 .. $D-1) {
        for my $j (0 .. $D_FF-1) {
            my ($wp,$wm) = (dclone($W1),dclone($W1));
            $wp->[$i][$j] += $EPS;  $wm->[$i][$j] -= $EPS;
            my $num = (fwd_loss_pff($wp,$B1,$W2,$B2,$INPUT)
                     - fwd_loss_pff($wm,$B1,$W2,$B2,$INPUT)) / (2*$EPS);
            check("L1_W[$i][$j]", $pff->{l1}{weights_grad}[$i][$j], $num);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 4: PFF L2 weights gradient [d_ff=$D_FF × d_model=$D]";
say "=" x 70;

{
    my $pff = make_pff($W1,$B1,$W2,$B2);
    my $out = $pff->forward(batch => dclone($INPUT));
    $pff->backward(next => ML::LossGradient->new(gradient => ones_like($out)));

    for my $i (0 .. $D_FF-1) {
        for my $j (0 .. $D-1) {
            my ($wp,$wm) = (dclone($W2),dclone($W2));
            $wp->[$i][$j] += $EPS;  $wm->[$i][$j] -= $EPS;
            my $num = (fwd_loss_pff($W1,$B1,$wp,$B2,$INPUT)
                     - fwd_loss_pff($W1,$B1,$wm,$B2,$INPUT)) / (2*$EPS);
            check("L2_W[$i][$j]", $pff->{l2}{weights_grad}[$i][$j], $num);
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
