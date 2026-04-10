#!/usr/bin/perl
# Probe higher learning rate with inline accuracy checks every 10 epochs
use Modern::Perl;
use lib '.';
use ML::Transformer; use ML::InputEmbeddings; use ML::PositionalEmbeddings;
use ML::Encoder; use ML::Decoder; use ML::Linear; use ML::DataLoader;
use ML::Util qw(create_src_mask create_tgt_mask softmax);

my $PAD="[PAD]"; my $SOS="[SOS]"; my $EOS="[EOS]";
my @C=("A","B","C"); my $V=[$PAD,$SOS,$EOS,@C];
my (%t2i,%i2t);
for my $i (0..$#$V){ $t2i{$V->[$i]}=$i; $i2t{$i}=$V->[$i]; }
my $VS=scalar @$V; my $PI=$t2i{$PAD}; my $EI=$t2i{$EOS}; my $SI=$t2i{$SOS};
my ($D,$NL,$NH,$DFF,$DR,$MPL)=(12,2,4,48,"0.01",5);
my $LR=0.05;

srand(1);
sub build {
    my $se=ML::InputEmbeddings->new(embeddings=>$D,vocab_size=>$VS);
    my $te=ML::InputEmbeddings->new(embeddings=>$D,vocab_size=>$VS);
    my $sp=ML::PositionalEmbeddings->new(embeddings=>$D,max_seq_length=>$MPL,dropout=>$DR);
    my $tp=ML::PositionalEmbeddings->new(embeddings=>$D,max_seq_length=>$MPL,dropout=>$DR);
    my $enc=ML::Encoder->new(embeddings=>$D,layers=>$NL,max_seq_length=>$MPL,dropout=>$DR,num_heads=>$NH,d_ff=>$DFF);
    my $dec=ML::Decoder->new(embeddings=>$D,layers=>$NL,max_seq_length=>$MPL,tgt_seq_length=>$MPL,dropout=>$DR,num_heads=>$NH,d_ff=>$DFF);
    my $pr=ML::Linear->new(insize=>$D,outsize=>$VS); $pr->set_weights_and_biases();
    return ML::Transformer->new(encoder=>$enc,decoder=>$dec,src_embed=>$se,tgt_embed=>$te,
                                src_pos=>$sp,tgt_pos=>$tp,projection_layer=>$pr);
}
sub gen { my ($n,$s)=@_; srand($s); my @d;
    for (1..$n){ my $l=2+int(rand(2)); my @q=map{$C[int rand 3]}1..$l;
        push @d,{src=>[$SOS,@q,$EOS],tgt=>[$SOS,@q,$EOS]}; } return \@d; }

sub tok_pad { my @ids=map{$t2i{$_}}@{$_[0]};
    push @ids,($PI) while @ids < $MPL; return [@ids[0..$MPL-1]]; }

sub greedy {
    my ($m,$src)=@_; my $sb=[$src];
    my $sm=create_src_mask(src_ids=>$sb,pad_id=>$PI);
    my $enc=$m->encode(src=>$sb,src_mask=>$sm);
    my $di=[[$SI,($PI)x($MPL-1)]];
    my @out;
    for my $t (0..$MPL-2){
        my $tm=create_tgt_mask(tgt_ids=>$di,pad_id=>$PI);
        my $dec=$m->decode(tgt=>$di,encoder_output=>$enc,src_mask=>$sm,tgt_mask=>$tm);
        my $log=$m->project(batch=>$dec);
        my $row=$log->[0][$t]; my $best=0;
        for my $v (1..$#$row){ $best=$v if $row->[$v]>$row->[$best]; }
        push @out,$best; last if $best==$EI;
        $di->[0][$t+1]=$best if $t+1<$MPL;
    }
    return \@out;
}

my $model=build();
my $loader=ML::DataLoader->new(data=>gen(80,1),token_to_id=>\%t2i,max_len=>$MPL,pad_id=>$PI);
my $test_data=gen(20,99);

for my $ep (1..50){
    $loader->init_iter(batch_size=>8);
    my $tot=0;
    for my $bn (0..$loader->{batches}-1){
        my $bt=$loader->next_batch();
        my $si=$bt->{src_ids}; my $di=$bt->{decoder_input_ids}; my $li=$bt->{label_ids};
        my $sm=create_src_mask(src_ids=>$si,pad_id=>$PI);
        my $tm=create_tgt_mask(tgt_ids=>$di,pad_id=>$PI);
        my $enc=$model->encode(src=>$si,src_mask=>$sm);
        my $dec=$model->decode(tgt=>$di,encoder_output=>$enc,src_mask=>$sm,tgt_mask=>$tm);
        my $log=$model->project(batch=>$dec);
        my ($fl,$fll)=([],[]); my $bs=8;
        my $sl=scalar @{$log->[0]}; my $vs=scalar @{$log->[0][0]};
        for my $b (0..$bs-1){ for my $s (0..$sl-1){
            for my $v (0..$vs-1){ $fl->[$b*$sl+$s][$v]=$log->[$b][$s][$v]; }
            $fll->[$b*$sl+$s]=$li->[$b][$s];
        }}
        my $sft=softmax($fl,1); my $loss=0; my $sc=$bs*$sl; my $lg=[];
        for my $i (0..$sc-1){ for my $j (0..$vs-1){
            if ($j==$fll->[$i]){ $loss+=-log($sft->[$i][$j])/$sc; $lg->[$i][$j]=$sft->[$i][$j]-1; }
            else { $lg->[$i][$j]=$sft->[$i][$j]; }
        }}
        $tot+=$loss;
        my $g=[];
        for my $b (0..$bs-1){ for my $s (0..$sl-1){ for my $v (0..$vs-1){
            $g->[$b][$s][$v]=$lg->[$b*$sl+$s][$v];
        }}}
        $model->update(projection=>1,decode=>1,encode=>1,gradient=>$g,learning_rate=>$LR);
    }
    my $avg=$tot/$loader->{batches};
    if ($ep%10==0 or $ep<=5){
        my ($ok,$n)=(0,0);
        for my $item (@$test_data){
            my $src=tok_pad($item->{src});
            my @exp=grep { $_ ne $SOS and $_ ne $EOS and $_ ne $PAD } @{$item->{tgt}};
            my $pred=greedy($model,$src);
            my @pc=map{$i2t{$_}} grep{$_!=$EI and $_!=$SI and $_!=$PI} @$pred;
            $ok++ if join(",",@pc) eq join(",",@exp);
            $n++;
        }
        printf "Epoch %3d | loss %.4f | acc %d/%d\n",$ep,$avg,$ok,$n;
    } else {
        printf "Epoch %3d | loss %.4f\n",$ep,$avg;
    }
}
