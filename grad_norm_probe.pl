#!/usr/bin/perl
# grad_norm_probe.pl — report gradient norms at every layer after one backward pass.
# Uses the small model (d=12) for speed.  Run with: perl grad_norm_probe.pl
#
# What to look for:
#   Healthy   : all norms within ~10x of each other across layers
#   Vanishing : encoder layer 0 << decoder layer N-1 (gradient dying before reaching early layers)
#   LayerNorm bottleneck: LN dgamma/dbeta very small compared to MHA dW norms
#   Cross-attn silent: cross-attn dWk/dWv ≈ 0 (gradient_kv channel broken)

use Modern::Perl;
use lib '.';
use ML::Transformer; use ML::InputEmbeddings; use ML::PositionalEmbeddings;
use ML::Encoder; use ML::Decoder; use ML::Linear; use ML::LossGradient;
use ML::Util qw(create_src_mask create_tgt_mask softmax);
use POSIX qw(sqrt);

# ── vocab (same as test_convergence.pl) ───────────────────────────────────────
my $PAD='[PAD]'; my $SOS='[SOS]'; my $EOS='[EOS]';
my @CONTENT=('A','B','C');
my $VOCAB=[$PAD,$SOS,$EOS,@CONTENT];
my (%tok2id,%id2tok);
for my $i (0..$#$VOCAB){ $tok2id{$VOCAB->[$i]}=$i; $id2tok{$i}=$VOCAB->[$i]; }
my $VOCAB_SIZE=scalar @$VOCAB;
my $PAD_ID=$tok2id{$PAD};

# ── model config ──────────────────────────────────────────────────────────────
my $D_MODEL    = 12;
my $NUM_LAYERS = 2;
my $NUM_HEADS  = 4;
my $D_FF       = 48;
my $DROPOUT    = "0.01";
my $MAX_PAD_LEN = 5;
my $BS         = 8;

# ── helpers ───────────────────────────────────────────────────────────────────
sub mat_norm {
    my ($m) = @_;
    return 0 unless defined $m && @$m && defined $m->[0] && ref $m->[0] eq 'ARRAY';
    my $sum = 0;
    for my $row (@$m) { $sum += $_ * $_ for @$row; }
    return sqrt($sum);
}

sub vec_norm {
    my ($v) = @_;
    return 0 unless defined $v && @$v;
    my $sum = 0;
    $sum += $_ * $_ for @$v;
    return sqrt($sum);
}

# ── build model ───────────────────────────────────────────────────────────────
srand(1);
my $se=ML::InputEmbeddings->new(embeddings=>$D_MODEL,vocab_size=>$VOCAB_SIZE);
my $te=ML::InputEmbeddings->new(embeddings=>$D_MODEL,vocab_size=>$VOCAB_SIZE);
my $sp=ML::PositionalEmbeddings->new(embeddings=>$D_MODEL,max_seq_length=>$MAX_PAD_LEN,dropout=>$DROPOUT);
my $tp=ML::PositionalEmbeddings->new(embeddings=>$D_MODEL,max_seq_length=>$MAX_PAD_LEN,dropout=>$DROPOUT);
my $enc=ML::Encoder->new(embeddings=>$D_MODEL,layers=>$NUM_LAYERS,max_seq_length=>$MAX_PAD_LEN,
                          dropout=>$DROPOUT,num_heads=>$NUM_HEADS,d_ff=>$D_FF);
my $dec=ML::Decoder->new(embeddings=>$D_MODEL,layers=>$NUM_LAYERS,max_seq_length=>$MAX_PAD_LEN,
                          tgt_seq_length=>$MAX_PAD_LEN,dropout=>$DROPOUT,num_heads=>$NUM_HEADS,d_ff=>$D_FF);
my $proj=ML::Linear->new(insize=>$D_MODEL,outsize=>$VOCAB_SIZE);
$proj->set_weights_and_biases();
my $model=ML::Transformer->new(encoder=>$enc,decoder=>$dec,src_embed=>$se,tgt_embed=>$te,
                                src_pos=>$sp,tgt_pos=>$tp,projection_layer=>$proj);

# ── one synthetic batch ───────────────────────────────────────────────────────
srand(42);
my (@batch_src,@batch_dec_in,@batch_labels);
for my $b (0..$BS-1) {
    my $len=2+int(rand(2));
    my @seq=map { $CONTENT[int rand 3] } 1..$len;
    my @src_ids   = map { $tok2id{$_} } ($SOS,@seq,$EOS);
    my @dec_ids   = map { $tok2id{$_} } ($SOS,@seq);
    my @label_ids = map { $tok2id{$_} } (@seq,$EOS);
    push @src_ids,   $PAD_ID while @src_ids   < $MAX_PAD_LEN;
    push @dec_ids,   $PAD_ID while @dec_ids   < $MAX_PAD_LEN;
    push @label_ids, $PAD_ID while @label_ids < $MAX_PAD_LEN;
    push @batch_src,    [@src_ids[0..$MAX_PAD_LEN-1]];
    push @batch_dec_in, [@dec_ids[0..$MAX_PAD_LEN-1]];
    push @batch_labels, [@label_ids[0..$MAX_PAD_LEN-1]];
}

# ── forward pass ──────────────────────────────────────────────────────────────
my $src_mask=create_src_mask(src_ids=>\@batch_src,   pad_id=>$PAD_ID);
my $tgt_mask=create_tgt_mask(tgt_ids=>\@batch_dec_in,pad_id=>$PAD_ID);
my $enc_out=$model->encode(src=>\@batch_src,src_mask=>$src_mask);
my $dec_out=$model->decode(tgt=>\@batch_dec_in,encoder_output=>$enc_out,
                            src_mask=>$src_mask,tgt_mask=>$tgt_mask);
my $logits=$model->project(batch=>$dec_out);

# ── loss gradient ─────────────────────────────────────────────────────────────
my $seq_len   = scalar @{$logits->[0]};
my $vocab_size= scalar @{$logits->[0][0]};
my $scale     = $BS * $seq_len;
my ($flat_logits,$flat_labels)=([],[]);
for my $b (0..$BS-1) { for my $s (0..$seq_len-1) {
    for my $v (0..$vocab_size-1) { $flat_logits->[$b*$seq_len+$s][$v]=$logits->[$b][$s][$v]; }
    $flat_labels->[$b*$seq_len+$s]=$batch_labels[$b][$s];
}}
my $sm=softmax($flat_logits,1);
my $loss_grad=[];
for my $i (0..$scale-1) { for my $j (0..$vocab_size-1) {
    $loss_grad->[$i][$j]=$sm->[$i][$j]-($j==$flat_labels->[$i]?1:0);
}}
my $grad=[];
for my $b (0..$BS-1) { for my $s (0..$seq_len-1) { for my $v (0..$vocab_size-1) {
    $grad->[$b][$s][$v]=$loss_grad->[$b*$seq_len+$s][$v];
}}}

# ── backward pass only (no optimise — preserves gradient arrays) ──────────────
my $next=ML::LossGradient->new(gradient=>$grad);
$model->{projection_layer}->backward(next=>$next);
$next=$model->{projection_layer};
$model->{decoder}->backward(next=>$next);
$model->{tgt_pos}->backward(next=>$model->{decoder});
$model->{tgt_embed}->backward(next=>$model->{tgt_pos});
my $enc_grad_arr=$model->{decoder}->gradient_enc();
my $enc_source=defined($enc_grad_arr)
    ? ML::LossGradient->new(gradient=>$enc_grad_arr)
    : $model->{tgt_embed};
$model->{encoder}->backward(next=>$enc_source);
$model->{src_pos}->backward(next=>$model->{encoder});
$model->{src_embed}->backward(next=>$model->{src_pos});

# ── report ────────────────────────────────────────────────────────────────────
say "=" x 70;
say "GRADIENT NORMS  (Frobenius for weight matrices, L2 for vectors)";
say "=" x 70;

say "\nPROJECTION LAYER";
printf "  weights_grad=%.4f  bias_grad=%.4f\n",
    mat_norm($model->{projection_layer}->{weights_grad}),
    mat_norm($model->{projection_layer}->{bias_grad});

say "\nENCODER";
for my $l (0..$NUM_LAYERS-1) {
    my $blk=$model->{encoder}->{EncoderBlocks}[$l];
    my $mha=$blk->{attention_block};
    my $ln0=$blk->{residual_connections}[0]->{ln_layer};
    my $ln1=$blk->{residual_connections}[1]->{ln_layer};
    say "  --- Layer $l ---";
    printf "  self-attn  dWq=%.4f  dWk=%.4f  dWv=%.4f  dWo=%.4f\n",
        mat_norm($mha->{dWq}), mat_norm($mha->{dWk}),
        mat_norm($mha->{dWv}), mat_norm($mha->{dWo});
    printf "  LN[attn]   dgamma=%.4f  dbeta=%.4f\n",
        vec_norm($ln0->{dgamma}), vec_norm($ln0->{dbeta});
    printf "  LN[ffn]    dgamma=%.4f  dbeta=%.4f\n",
        vec_norm($ln1->{dgamma}), vec_norm($ln1->{dbeta});
}

say "\nDECODER";
for my $l (0..$NUM_LAYERS-1) {
    my $blk=$model->{decoder}->{DecoderBlocks}[$l];
    my $sa=$blk->{self_attention_block};
    my $xa=$blk->{cross_attention_block};
    my $ln0=$blk->{residual_connections}[0]->{ln_layer};
    my $ln1=$blk->{residual_connections}[1]->{ln_layer};
    my $ln2=$blk->{residual_connections}[2]->{ln_layer};
    say "  --- Layer $l ---";
    printf "  self-attn  dWq=%.4f  dWk=%.4f  dWv=%.4f  dWo=%.4f\n",
        mat_norm($sa->{dWq}), mat_norm($sa->{dWk}),
        mat_norm($sa->{dWv}), mat_norm($sa->{dWo});
    printf "  cross-attn dWq=%.4f  dWk=%.4f  dWv=%.4f  dWo=%.4f\n",
        mat_norm($xa->{dWq}), mat_norm($xa->{dWk}),
        mat_norm($xa->{dWv}), mat_norm($xa->{dWo});
    printf "  LN[self]   dgamma=%.4f  dbeta=%.4f\n",
        vec_norm($ln0->{dgamma}), vec_norm($ln0->{dbeta});
    printf "  LN[cross]  dgamma=%.4f  dbeta=%.4f\n",
        vec_norm($ln1->{dgamma}), vec_norm($ln1->{dbeta});
    printf "  LN[ffn]    dgamma=%.4f  dbeta=%.4f\n",
        vec_norm($ln2->{dgamma}), vec_norm($ln2->{dbeta});
}

say "\n" . "=" x 70;
say "INTERPRETATION";
say "=" x 70;
say "  Healthy      : all norms within ~10x of each other";
say "  Vanishing    : encoder layer 0 << decoder layer " . ($NUM_LAYERS-1);
say "  LN bottleneck: LN dgamma/dbeta << MHA dW norms";
say "  Cross-attn   : dWk/dWv ≈ 0 means gradient_kv channel not flowing";
say "=" x 70;
