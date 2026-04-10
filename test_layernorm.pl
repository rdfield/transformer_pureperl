#!/usr/bin/perl
# test_layernorm.pl — gradient + Adam correctness check for ML::LayerNormalisation
#
# Uses central finite differences as ground truth (same approach as test_linear.pl
# and test_mha.pl).
#
# For each parameter θ: numerical_grad = (L(θ+ε) - L(θ-ε)) / 2ε
# Loss: L = sum of all output elements  →  dL/doutput = ones everywhere.
#
# Tests:
#   1. dgamma  — gradient w.r.t. scale parameter (one value per embedding dim)
#   2. dbeta   — gradient w.r.t. shift parameter (one value per embedding dim)
#   3. dinput  — gradient w.r.t. input [batch × seq × emb]
#   4. Adam step — one optimise() call produces the correct weight update

use Modern::Perl;
use lib '.';
use ML::LayerNormalisation;
use ML::LossGradient;
use Storable qw(dclone);
use POSIX qw(sqrt);

my $EPS = 1e-4;
my $TOL = 1e-3;
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
        printf "  PASS  %-40s  anal=%10.6f  num=%10.6f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $PASS++;
    } else {
        printf "  FAIL  %-40s  anal=%10.6f  num=%10.6f  err=%.2e\n",
            $label, $analytical, $numerical, $err;
        $FAIL++;
    }
}

sub ok {
    my ($label, $cond) = @_;
    if ($cond) { printf "  PASS  %s\n", $label; $PASS++ }
    else        { printf "  FAIL  %s\n", $label; $FAIL++ }
}

# ── sum all output elements (the loss L = sum(output)) ────────────────────────

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

# ── build and run a fresh LayerNorm with given gamma/beta/input ───────────────

sub make_ln {
    my ($EMB, $gamma, $beta) = @_;
    my $ln = ML::LayerNormalisation->new(embeddings => $EMB, max_seq_length => 10);
    $ln->initialise(weights => { gamma => dclone($gamma), beta => dclone($beta) });
    return $ln;
}

sub fwd_loss {
    my ($EMB, $gamma, $beta, $input) = @_;
    my $ln  = make_ln($EMB, $gamma, $beta);
    my $out = $ln->forward(batch => dclone($input));
    return sum_output($out);
}

# ── fixed test values ─────────────────────────────────────────────────────────

my $BATCH = 2;
my $SEQ   = 3;
my $EMB   = 4;

# gamma != 1 and beta != 0 so gradients are non-trivial
my $GAMMA = [1.2, -0.8, 0.5, 1.5];
my $BETA  = [0.1, -0.3, 0.2, -0.1];

my $INPUT = [
    [ [ 0.5, -1.2,  0.8,  0.3],
      [ 1.1,  0.3, -0.6,  0.9],
      [-0.4,  0.7,  0.2, -0.8] ],
    [ [-0.3,  0.9,  1.4, -0.5],
      [ 0.6, -0.8,  0.1,  1.2],
      [ 0.2,  0.5, -1.1,  0.4] ],
];

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: dgamma  [embeddings=$EMB]";
say "=" x 70;

{
    my $ln  = make_ln($EMB, $GAMMA, $BETA);
    my $out = $ln->forward(batch => dclone($INPUT));
    $ln->backward(next => ML::LossGradient->new(gradient => ones_3d($out)),
                  batch_size => $BATCH);

    for my $e (0 .. $EMB - 1) {
        my ($gp, $gm) = (dclone($GAMMA), dclone($GAMMA));
        $gp->[$e] += $EPS;  $gm->[$e] -= $EPS;
        my $num = (fwd_loss($EMB, $gp, $BETA, $INPUT)
                 - fwd_loss($EMB, $gm, $BETA, $INPUT)) / (2 * $EPS);
        check("dgamma[$e]", $ln->{dgamma}[$e], $num);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: dbeta  [embeddings=$EMB]";
say "=" x 70;

{
    my $ln  = make_ln($EMB, $GAMMA, $BETA);
    my $out = $ln->forward(batch => dclone($INPUT));
    $ln->backward(next => ML::LossGradient->new(gradient => ones_3d($out)),
                  batch_size => $BATCH);

    for my $e (0 .. $EMB - 1) {
        my ($bp, $bm) = (dclone($BETA), dclone($BETA));
        $bp->[$e] += $EPS;  $bm->[$e] -= $EPS;
        my $num = (fwd_loss($EMB, $GAMMA, $bp, $INPUT)
                 - fwd_loss($EMB, $GAMMA, $bm, $INPUT)) / (2 * $EPS);
        check("dbeta[$e]", $ln->{dbeta}[$e], $num);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 3: input gradient  [batch=$BATCH × seq=$SEQ × emb=$EMB]";
say "=" x 70;

{
    my $ln  = make_ln($EMB, $GAMMA, $BETA);
    my $out = $ln->forward(batch => dclone($INPUT));
    $ln->backward(next => ML::LossGradient->new(gradient => ones_3d($out)),
                  batch_size => $BATCH);
    my $grad = $ln->gradient();

    for my $b (0 .. $BATCH - 1) {
        for my $s (0 .. $SEQ - 1) {
            for my $e (0 .. $EMB - 1) {
                my ($ip, $im) = (dclone($INPUT), dclone($INPUT));
                $ip->[$b][$s][$e] += $EPS;  $im->[$b][$s][$e] -= $EPS;
                my $num = (fwd_loss($EMB, $GAMMA, $BETA, $ip)
                         - fwd_loss($EMB, $GAMMA, $BETA, $im)) / (2 * $EPS);
                check("dinput[$b][$s][$e]", $grad->[$b][$s][$e], $num);
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 4: Adam update — one optimise() step";
say "=" x 70;

# Verify that after one optimise() call, gamma and beta are updated exactly
# according to the Adam formula with t=1, beta1=0.9, beta2=0.999, eps=1e-8.
#
# With t=1:
#   m_hat = (1-beta1)*g / (1-beta1^1) = g        (bias correction cancels)
#   v_hat = (1-beta2)*g^2 / (1-beta2^1) = g^2    (bias correction cancels)
#   update = lr * g / (|g| + eps)
# For g=1.0, lr=0.001: update ≈ 0.001 (to within eps=1e-8)
{
    my $LR    = 0.001;
    my $beta1 = 0.9;
    my $beta2 = 0.999;
    my $eps   = 1e-8;

    # Use simple gamma=1, beta=0, known dgamma/dbeta after one backward
    my $ln  = make_ln($EMB, $GAMMA, $BETA);
    my $out = $ln->forward(batch => dclone($INPUT));
    $ln->backward(next => ML::LossGradient->new(gradient => ones_3d($out)),
                  batch_size => $BATCH);

    # Snapshot dgamma/dbeta before optimise clears them
    my @dgamma = @{$ln->{dgamma}};
    my @dbeta  = @{$ln->{dbeta}};
    my @gamma0 = @{$ln->{gamma}};
    my @beta0  = @{$ln->{beta}};

    $ln->optimise(learning_rate => $LR);

    # t=1, moments start at zero → first Adam step
    for my $e (0 .. $EMB - 1) {
        my $g = $dgamma[$e];
        my $m = (1 - $beta1) * $g;
        my $v = (1 - $beta2) * $g ** 2;
        my $m_hat = $m / (1 - $beta1);   # = g
        my $v_hat = $v / (1 - $beta2);   # = g^2
        my $expected = $gamma0[$e] - $LR * $m_hat / (sqrt($v_hat) + $eps);
        check("gamma[$e] after Adam step", $ln->{gamma}[$e], $expected);

        $g = $dbeta[$e];
        $m = (1 - $beta1) * $g;
        $v = (1 - $beta2) * $g ** 2;
        $m_hat = $m / (1 - $beta1);
        $v_hat = $v / (1 - $beta2);
        $expected = $beta0[$e] - $LR * $m_hat / (sqrt($v_hat) + $eps);
        check("beta[$e]  after Adam step", $ln->{beta}[$e],  $expected);
    }

    # adam_epoch should have incremented to 2
    ok("adam_epoch incremented to 2 after one optimise()", $ln->{adam_epoch} == 2);

    # dgamma and dbeta should be zeroed after optimise()
    my $all_zero = 1;
    for my $e (0 .. $EMB - 1) {
        $all_zero = 0 if $ln->{dgamma}[$e] != 0 || $ln->{dbeta}[$e] != 0;
    }
    ok("dgamma and dbeta zeroed after optimise()", $all_zero);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
