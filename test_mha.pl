#!/usr/bin/perl
# test_mha.pl — gradient check for ML::MHA
#
# Uses central finite differences as ground truth for every gradient.
# Loss: L = sum of all output elements  →  dL/doutput = ones everywhere.
#
# Tests:
#   1. Self-attention  (batch only — q=k=v=input)
#      Checks: dWq, dWk, dWv, dWo, d_input (gradient)
#   2. Cross-attention (batch=Q, batch_1=batch_2=K/V source)
#      Checks: dWq, dWk, dWv, dWo, d_Q (gradient), d_KV (gradient_kv)
#
# dropout is set to "0.0" so attention weights are deterministic.

use Modern::Perl;
use lib '.';
use ML::MHA;
use ML::LossGradient;
use Storable qw(dclone);
use POSIX qw(sqrt);

my $EPS = 1e-4;
my $TOL = 1e-3;
my ($PASS, $FAIL) = (0, 0);
my @FAILURES;

# ── model dimensions ──────────────────────────────────────────────────────────
my $EMBED = 8;
my $HEADS = 2;
my $MSL   = 10;   # max_seq_length (must be >= SEQ)
my $SEQ   = 3;
my $BATCH = 2;

# ── helpers ───────────────────────────────────────────────────────────────────
sub rel_error {
    my ($a, $b) = @_;
    my $d = abs($a) > abs($b) ? abs($a) : abs($b);
    return 0 if $d < 1e-12;
    return abs($a - $b) / $d;
}

sub tally {
    my ($label, $anal, $num) = @_;
    my $err = rel_error($anal, $num);
    if ($err < $TOL) { $PASS++; }
    else {
        $FAIL++;
        push @FAILURES, sprintf("  FAIL  %-45s  anal=%10.6f  num=%10.6f  err=%.2e",
            $label, $anal, $num, $err);
    }
}

sub section_header { say "\n" . $_[0] }

sub sum_3d {
    my ($t) = @_;
    my $s = 0;
    for my $b (@$t) { for my $r (@$b) { $s += $_ for @$r } }
    return $s;
}

sub ones_3d {
    my ($b, $s, $e) = @_;
    my $t = [];
    for my $bi (0..$b-1) { for my $si (0..$s-1) { for my $ei (0..$e-1) {
        $t->[$bi][$si][$ei] = 1.0;
    }}}
    return $t;
}

sub make_mha {
    my ($Wq, $Wk, $Wv, $Wo) = @_;
    return ML::MHA->new(
        embeddings     => $EMBED,
        num_heads      => $HEADS,
        max_seq_length => $MSL,
        dropout        => "0.0",
        W_q => dclone($Wq),
        W_k => dclone($Wk),
        W_v => dclone($Wv),
        W_o => dclone($Wo),
    );
}

sub fwd_self {
    my ($Wq, $Wk, $Wv, $Wo, $inp) = @_;
    my $m = make_mha($Wq, $Wk, $Wv, $Wo);
    return sum_3d($m->forward(batch => dclone($inp)));
}

sub fwd_cross {
    my ($Wq, $Wk, $Wv, $Wo, $q, $kv) = @_;
    my $m = make_mha($Wq, $Wk, $Wv, $Wo);
    return sum_3d($m->forward(
        batch   => dclone($q),
        batch_1 => dclone($kv),
        batch_2 => dclone($kv),
    ));
}

sub check_weight_matrix {
    my ($name, $W_ref, $dW, $fwd_fn, $other_args) = @_;
    # $other_args is an arrayref of remaining args after W matrices
    section_header("  $name  [$EMBED x $EMBED = " . ($EMBED*$EMBED) . " elements]");
    my $before = $FAIL;
    for my $i (0 .. $EMBED-1) {
        for my $j (0 .. $EMBED-1) {
            my ($Wp, $Wm) = (dclone($W_ref), dclone($W_ref));
            $Wp->[$i][$j] += $EPS;
            $Wm->[$i][$j] -= $EPS;
            my $num = ($fwd_fn->($Wp, @$other_args) - $fwd_fn->($Wm, @$other_args)) / (2*$EPS);
            tally("$name\[$i\]\[$j\]", $dW->[$i][$j], $num);
        }
    }
    my $new_fails = $FAIL - $before;
    my $n = $EMBED * $EMBED;
    printf "    %d/%d pass%s\n", $n - $new_fails, $n, ($new_fails ? "  *** $new_fails FAIL(S) ***" : "");
}

sub check_input_3d {
    my ($name, $grad, $inp_ref, $fwd_fn, $other_args) = @_;
    section_header("  $name  [$BATCH x $SEQ x $EMBED]");
    my $before = $FAIL;
    for my $b (0 .. $BATCH-1) {
        for my $s (0 .. $SEQ-1) {
            for my $e (0 .. $EMBED-1) {
                my ($ip, $im) = (dclone($inp_ref), dclone($inp_ref));
                $ip->[$b][$s][$e] += $EPS;
                $im->[$b][$s][$e] -= $EPS;
                my $num = ($fwd_fn->($ip, @$other_args) - $fwd_fn->($im, @$other_args)) / (2*$EPS);
                tally("$name\[$b\]\[$s\]\[$e\]", $grad->[$b][$s][$e], $num);
            }
        }
    }
    my $new_fails = $FAIL - $before;
    my $n = $BATCH * $SEQ * $EMBED;
    printf "    %d/%d pass%s\n", $n - $new_fails, $n, ($new_fails ? "  *** $new_fails FAIL(S) ***" : "");
}

# ── fixed test data (seeded for reproducibility) ─────────────────────────────
srand(42);
sub rand_mat { my($r,$c)=@_; my $m=[];
    for my $i(0..$r-1){for my $j(0..$c-1){ $m->[$i][$j]=(rand()-0.5)*0.4; }} return $m; }
sub rand_inp { my($b,$s,$e)=@_; my $t=[];
    for my $bi(0..$b-1){for my $si(0..$s-1){for my $ei(0..$e-1){
        $t->[$bi][$si][$ei]=(rand()-0.5)*0.4; }}} return $t; }

my $Wq  = rand_mat($EMBED, $EMBED);
my $Wk  = rand_mat($EMBED, $EMBED);
my $Wv  = rand_mat($EMBED, $EMBED);
my $Wo  = rand_mat($EMBED, $EMBED);
my $inp = rand_inp($BATCH, $SEQ, $EMBED);
my $q   = rand_inp($BATCH, $SEQ, $EMBED);
my $kv  = rand_inp($BATCH, $SEQ, $EMBED);

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: SELF-ATTENTION  [batch=$BATCH  seq=$SEQ  embed=$EMBED  heads=$HEADS]";
say "  q = k = v = input  (is_cross_attention = 0)";
say "=" x 70;

{
    my $mha = make_mha($Wq, $Wk, $Wv, $Wo);
    $mha->forward(batch => dclone($inp));
    $mha->backward(next => ML::LossGradient->new(
        gradient => ones_3d($BATCH, $SEQ, $EMBED)));

    # Weight gradients — perturb each weight, rerun forward, finite diff
    check_weight_matrix("dWq", $Wq, $mha->{dWq},
        sub { fwd_self(shift, $Wk, $Wv, $Wo, $inp) }, []);
    check_weight_matrix("dWk", $Wk, $mha->{dWk},
        sub { fwd_self($Wq, shift, $Wv, $Wo, $inp) }, []);
    check_weight_matrix("dWv", $Wv, $mha->{dWv},
        sub { fwd_self($Wq, $Wk, shift, $Wo, $inp) }, []);
    check_weight_matrix("dWo", $Wo, $mha->{dWo},
        sub { fwd_self($Wq, $Wk, $Wv, shift, $inp) }, []);

    # Input gradient — perturb input, rerun forward
    check_input_3d("d_input", $mha->{gradient}, $inp,
        sub { fwd_self($Wq, $Wk, $Wv, $Wo, shift) }, []);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: CROSS-ATTENTION  [batch=$BATCH  seq=$SEQ  embed=$EMBED  heads=$HEADS]";
say "  Q = batch,  K = V = batch_1 = batch_2  (is_cross_attention = 1)";
say "=" x 70;

{
    my $mha = make_mha($Wq, $Wk, $Wv, $Wo);
    $mha->forward(batch => dclone($q), batch_1 => dclone($kv), batch_2 => dclone($kv));
    $mha->backward(next => ML::LossGradient->new(
        gradient => ones_3d($BATCH, $SEQ, $EMBED)));

    check_weight_matrix("dWq", $Wq, $mha->{dWq},
        sub { fwd_cross(shift, $Wk, $Wv, $Wo, $q, $kv) }, []);
    check_weight_matrix("dWk", $Wk, $mha->{dWk},
        sub { fwd_cross($Wq, shift, $Wv, $Wo, $q, $kv) }, []);
    check_weight_matrix("dWv", $Wv, $mha->{dWv},
        sub { fwd_cross($Wq, $Wk, shift, $Wo, $q, $kv) }, []);
    check_weight_matrix("dWo", $Wo, $mha->{dWo},
        sub { fwd_cross($Wq, $Wk, $Wv, shift, $q, $kv) }, []);

    # Q gradient (decoder stream) — lives in gradient
    check_input_3d("d_Q (gradient)", $mha->{gradient}, $q,
        sub { fwd_cross($Wq, $Wk, $Wv, $Wo, shift, $kv) }, []);

    # K/V gradient (encoder stream) — lives in gradient_kv
    check_input_3d("d_KV (gradient_kv)", $mha->{gradient_kv}, $kv,
        sub { fwd_cross($Wq, $Wk, $Wv, $Wo, $q, shift) }, []);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
if (@FAILURES) {
    say "FAILURES:";
    say $_ for @FAILURES;
    say "";
}
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
