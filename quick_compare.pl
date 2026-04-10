#!/usr/bin/perl
# Quick 15-epoch comparison: correct encoder_output key vs original encode_output (broken) key
use Modern::Perl;
use lib '.';
use ML::Transformer;
use ML::InputEmbeddings;
use ML::PositionalEmbeddings;
use ML::Encoder;
use ML::Decoder;
use ML::Linear;
use ML::DataLoader;
use ML::Util qw(create_src_mask create_tgt_mask softmax);

my $PAD = '[PAD]'; my $SOS = '[SOS]'; my $EOS = '[EOS]';
my @CONTENT = ('A','B','C');
my $VOCAB   = [$PAD,$SOS,$EOS,@CONTENT];
my (%t2i,%i2t);
for my $i (0..$#$VOCAB){ $t2i{$VOCAB->[$i]}=$i; $i2t{$i}=$VOCAB->[$i]; }
my $VS=scalar @$VOCAB; my $PI=$t2i{$PAD};

my $D=12; my $NL=2; my $NH=4; my $DFF=48; my $DR="0.01"; my $MPL=5;

sub build_model {
    my $se=ML::InputEmbeddings->new(embeddings=>$D,vocab_size=>$VS);
    my $te=ML::InputEmbeddings->new(embeddings=>$D,vocab_size=>$VS);
    my $sp=ML::PositionalEmbeddings->new(embeddings=>$D,max_seq_length=>$MPL,dropout=>$DR);
    my $tp=ML::PositionalEmbeddings->new(embeddings=>$D,max_seq_length=>$MPL,dropout=>$DR);
    my $enc=ML::Encoder->new(embeddings=>$D,layers=>$NL,max_seq_length=>$MPL,dropout=>$DR,num_heads=>$NH,d_ff=>$DFF);
    my $dec=ML::Decoder->new(embeddings=>$D,layers=>$NL,max_seq_length=>$MPL,tgt_seq_length=>$MPL,dropout=>$DR,num_heads=>$NH,d_ff=>$DFF);
    my $pr=ML::Linear->new(insize=>$D,outsize=>$VS);
    $pr->set_weights_and_biases();
    return ML::Transformer->new(encoder=>$enc,decoder=>$dec,src_embed=>$se,tgt_embed=>$te,
                                src_pos=>$sp,tgt_pos=>$tp,projection_layer=>$pr);
}

sub gen_data {
    my ($n)=@_; my @d;
    for (1..$n){
        my $l=2+int(rand(2));
        my @s=map{$CONTENT[int rand 3]}1..$l;
        push @d,{src=>[$SOS,@s,$EOS],tgt=>[$SOS,@s,$EOS]};
    }
    return \@d;
}

# Use the same shuffled data for both runs for fair comparison
srand(42);
my $data=gen_data(80);  # 10 batches/epoch → ~3s/epoch

sub run_epochs {
    my ($label, $use_correct_key) = @_;
    srand(42);
    my $model = build_model();
    my $loader = ML::DataLoader->new(data=>$data, token_to_id=>\%t2i,
                                     max_len=>$MPL, pad_id=>$PI);
    printf "\n=== %s (key: %s) ===\n", $label,
           $use_correct_key ? "encoder_output (correct)" : "encode_output (original/broken)";
    for my $ep (1..25) {
        $loader->init_iter(batch_size=>8);
        my $total=0;
        for my $bn (0..$loader->{batches}-1) {
            my $batch=$loader->next_batch();
            my $si=$batch->{src_ids}; my $di=$batch->{decoder_input_ids}; my $li=$batch->{label_ids};
            my $sm=create_src_mask(src_ids=>$si, pad_id=>$PI);
            my $tm=create_tgt_mask(tgt_ids=>$di, pad_id=>$PI);
            my $enc=$model->encode(src=>$si, src_mask=>$sm);
            my $dec;
            if ($use_correct_key) {
                $dec=$model->decode(tgt=>$di, encoder_output=>$enc, src_mask=>$sm, tgt_mask=>$tm);
            } else {
                $dec=$model->decode(tgt=>$di, encode_output=>$enc,  src_mask=>$sm, tgt_mask=>$tm);
            }
            my $log=$model->project(batch=>$dec);
            my $bs=8; my $sl=scalar @{$log->[0]}; my $vs=scalar @{$log->[0][0]};
            my ($fl,$fll)=([],[]);
            for my $b(0..$bs-1){ for my $s(0..$sl-1){
                for my $v(0..$vs-1){$fl->[$b*$sl+$s][$v]=$log->[$b][$s][$v];}
                $fll->[$b*$sl+$s]=$li->[$b][$s];
            }}
            my $sft=softmax($fl,1); my $loss=0; my $sc=$bs*$sl; my $lg=[];
            for my $i(0..$sc-1){ for my $j(0..$vs-1){
                if($j==$fll->[$i]){$loss+=-log($sft->[$i][$j])/$sc; $lg->[$i][$j]=$sft->[$i][$j]-1;}
                else{$lg->[$i][$j]=$sft->[$i][$j];}
            }}
            $total+=$loss;
            my $g=[];
            for my $b(0..$bs-1){ for my $s(0..$sl-1){ for my $v(0..$vs-1){
                $g->[$b][$s][$v]=$lg->[$b*$sl+$s][$v];
            }}}
            $model->update(projection=>1,decode=>1,encode=>1,gradient=>$g,learning_rate=>0.01);
        }
        printf "  Epoch %2d | loss %.4f\n", $ep, $total/$loader->{batches};
    }
}

run_epochs("CORRECT cross-attention",  1);
run_epochs("ORIGINAL decoder-only",    0);
