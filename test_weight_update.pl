#!/usr/bin/perl
# test_weight_update.pl
#
# Runs 1 epoch (2 batches of 2) on a tiny copy-task dataset.
# Saves initial and final weights as formatted JSON files for
# side-by-side diff on jsondiff.com.
#
# Output files: twu_YYYYMMDD_HHMMSS_init.json
#               twu_YYYYMMDD_HHMMSS_final.json
#
# Format: keys sorted alphabetically, each weight/bias row on one line.

use Modern::Perl;
use lib '.';
use POSIX qw(strftime);
use Scalar::Util qw(looks_like_number);

use ML::Transformer;
use ML::InputEmbeddings;
use ML::PositionalEmbeddings;
use ML::Encoder;
use ML::Decoder;
use ML::Linear;
use ML::DataLoader;
use ML::Util qw(create_src_mask create_tgt_mask softmax);

# ── parameters ────────────────────────────────────────────────────────────────
my $D_MODEL    = 12;
my $NUM_HEADS  = 2;
my $NUM_LAYERS = 2;
my $D_FF       = 48;   # 4 × d_model
my $DROPOUT    = 0.1;
my $BATCH_SIZE = 2;
my $LR         = 0.001;

my $PAD_TOKEN = '[PAD]';
my $SOS_TOKEN = '[SOS]';
my $EOS_TOKEN = '[EOS]';
my @CONTENT   = ('A', 'B', 'C');
my @VOCAB     = ($PAD_TOKEN, $SOS_TOKEN, $EOS_TOKEN, @CONTENT);

my %token_to_id; my %id_to_token;
for my $i (0 .. $#VOCAB) {
    $token_to_id{$VOCAB[$i]} = $i;
    $id_to_token{$i}         = $VOCAB[$i];
}
my $VOCAB_SIZE = scalar @VOCAB;
my $PAD_ID     = $token_to_id{$PAD_TOKEN};
my $SOS_ID     = $token_to_id{$SOS_TOKEN};
my $EOS_ID     = $token_to_id{$EOS_TOKEN};

# 4 fixed training examples (2 tokens each → padded length 4: SOS + 2 + EOS)
my $MAX_SEQ_LEN    = 2;
my $MAX_PADDED_LEN = $MAX_SEQ_LEN + 2;

my @raw_data = (
    { src => [$SOS_TOKEN, 'A', 'B', $EOS_TOKEN], tgt => [$SOS_TOKEN, 'A', 'B', $EOS_TOKEN] },
    { src => [$SOS_TOKEN, 'B', 'C', $EOS_TOKEN], tgt => [$SOS_TOKEN, 'B', 'C', $EOS_TOKEN] },
    { src => [$SOS_TOKEN, 'C', 'A', $EOS_TOKEN], tgt => [$SOS_TOKEN, 'C', 'A', $EOS_TOKEN] },
    { src => [$SOS_TOKEN, 'A', 'C', $EOS_TOKEN], tgt => [$SOS_TOKEN, 'A', 'C', $EOS_TOKEN] },
);

# ── model construction ────────────────────────────────────────────────────────
my $src_embed = ML::InputEmbeddings->new(embeddings => $D_MODEL, vocab_size => $VOCAB_SIZE);
my $tgt_embed = ML::InputEmbeddings->new(embeddings => $D_MODEL, vocab_size => $VOCAB_SIZE);
my $src_pos   = ML::PositionalEmbeddings->new(embeddings => $D_MODEL, max_seq_length => $MAX_PADDED_LEN, dropout => $DROPOUT);
my $tgt_pos   = ML::PositionalEmbeddings->new(embeddings => $D_MODEL, max_seq_length => $MAX_PADDED_LEN, dropout => $DROPOUT);

my $encoder = ML::Encoder->new(
    embeddings     => $D_MODEL,
    layers         => $NUM_LAYERS,
    max_seq_length => $MAX_PADDED_LEN,
    dropout        => $DROPOUT,
    num_heads      => $NUM_HEADS,
    d_ff           => $D_FF,
);
my $decoder = ML::Decoder->new(
    embeddings     => $D_MODEL,
    layers         => $NUM_LAYERS,
    max_seq_length => $MAX_PADDED_LEN,
    dropout        => $DROPOUT,
    d_ff           => $D_FF,
    tgt_seq_length => $MAX_PADDED_LEN,
    num_heads      => $NUM_HEADS,
);
my $projection = ML::Linear->new(insize => $D_MODEL, outsize => $VOCAB_SIZE);
$projection->set_weights_and_biases();

my $model = ML::Transformer->new(
    encoder          => $encoder,
    decoder          => $decoder,
    src_embed        => $src_embed,
    tgt_embed        => $tgt_embed,
    src_pos          => $src_pos,
    tgt_pos          => $tgt_pos,
    projection_layer => $projection,
);

# ── helpers ───────────────────────────────────────────────────────────────────
sub tokenize_pad {
    my ($tokens) = @_;
    my @ids = map { $token_to_id{$_} } @$tokens;
    while (@ids < $MAX_PADDED_LEN) { push @ids, $PAD_ID }
    return [@ids[0 .. $MAX_PADDED_LEN - 1]];
}

# Custom JSON formatter: keys sorted alphabetically, leaf arrays on one line.
sub _fmt {
    my ($data, $depth) = @_;
    $depth //= 0;
    my $pad  = "  " x $depth;
    my $pad1 = "  " x ($depth + 1);

    if (!defined $data) {
        return 'null';
    } elsif (ref $data eq 'HASH') {
        return '{}' unless %$data;
        my @pairs;
        for my $k (sort keys %$data) {
            push @pairs, $pad1 . '"' . $k . '": ' . _fmt($data->{$k}, $depth + 1);
        }
        return "{\n" . join(",\n", @pairs) . "\n$pad}";
    } elsif (ref $data eq 'ARRAY') {
        return '[]' unless @$data;
        # If any element is a reference (HASH or ARRAY), expand one-element-per-line
        if (ref($data->[0])) {
            my @rows;
            for my $row (@$data) {
                push @rows, $pad1 . _fmt($row, $depth + 1);
            }
            return "[\n" . join(",\n", @rows) . "\n$pad]";
        } else {
            # Leaf array (numbers/strings) — all on one line
            my @vals = map {
                !defined($_)          ? 'null'
                : looks_like_number($_) ? $_
                : do { (my $s = $_) =~ s/"/\\"/g; "\"$s\"" }
            } @$data;
            return '[' . join(', ', @vals) . ']';
        }
    } elsif (looks_like_number($data)) {
        return $data;
    } else {
        (my $s = $data) =~ s/"/\\"/g;
        return "\"$s\"";
    }
}

sub write_formatted_json {
    my ($filename, $data) = @_;
    open my $fh, '>:encoding(utf-8)', $filename or die "Cannot write $filename: $!";
    print $fh _fmt($data) . "\n";
    close $fh;
    say "Written: $filename";
}

# ── save initial weights ──────────────────────────────────────────────────────
my $ts = strftime("%Y%m%d_%H%M%S", localtime);
my $init_file  = "twu_${ts}_init.json";
my $final_file = "twu_${ts}_final.json";

write_formatted_json($init_file, {
    src_embed        => $model->{src_embed}->get_weights(),
    tgt_embed        => $model->{tgt_embed}->get_weights(),
    encoder          => $model->{encoder}->get_weights(),
    decoder          => $model->{decoder}->get_weights(),
    projection_layer => $model->{projection_layer}->get_weights(),
});

# ── training loop (1 epoch, 2 batches of 2) ───────────────────────────────────
say "Training: 1 epoch, ${\scalar @raw_data} examples, batch_size=$BATCH_SIZE";

# Shuffle and split into batches manually (fixed order for reproducibility)
for my $batchno (0 .. 1) {
    my @batch_data = @raw_data[$batchno * $BATCH_SIZE .. ($batchno + 1) * $BATCH_SIZE - 1];

    my $src_ids          = [];
    my $decoder_input_ids = [];
    my $label_ids        = [];

    for my $b (0 .. $#batch_data) {
        $src_ids->[$b]           = tokenize_pad($batch_data[$b]{src});
        # decoder input: tgt shifted right (SOS + content, drop EOS)
        my @tgt_ids = map { $token_to_id{$_} } @{$batch_data[$b]{tgt}};
        $decoder_input_ids->[$b] = [@tgt_ids[0 .. $MAX_PADDED_LEN - 2], $PAD_ID];
        # labels: tgt shifted left (content + EOS, drop SOS) + trailing PAD to match seq_len
        $label_ids->[$b]         = [@tgt_ids[1 .. $MAX_PADDED_LEN - 1], $PAD_ID];
    }

    my $src_mask = create_src_mask(src_ids => $src_ids, pad_id => $PAD_ID);
    my $tgt_mask = create_tgt_mask(tgt_ids => $decoder_input_ids, pad_id => $PAD_ID);

    my $encoder_output = $model->encode(src => $src_ids, src_mask => $src_mask);
    my $decoder_output = $model->decode(
        tgt            => $decoder_input_ids,
        encoder_output => $encoder_output,
        src_mask       => $src_mask,
        tgt_mask       => $tgt_mask,
    );
    my $final_output = $model->project(batch => $decoder_output);

    my $seq_len   = scalar @{$final_output->[0]};
    my $vocab_sz  = scalar @{$final_output->[0][0]};
    my $flat_out  = [];
    my $flat_lab  = [];
    for my $b (0 .. $BATCH_SIZE - 1) {
        for my $s (0 .. $seq_len - 1) {
            for my $v (0 .. $vocab_sz - 1) {
                $flat_out->[$b * $seq_len + $s][$v] = $final_output->[$b][$s][$v];
            }
            $flat_lab->[$b * $seq_len + $s] = $label_ids->[$b][$s];
        }
    }

    my $sm         = softmax($flat_out, 1);
    my $n_tok      = scalar grep { $_ != $PAD_ID } @$flat_lab;
    my $loss_scale = $n_tok || 1;
    my $grad_scale = $BATCH_SIZE * $seq_len;
    my $loss       = 0;
    my $loss_grad  = [];

    for my $i (0 .. $BATCH_SIZE * $seq_len - 1) {
        if ($flat_lab->[$i] == $PAD_ID) {
            $loss_grad->[$i][$_] = 0 for 0 .. $vocab_sz - 1;
            next;
        }
        for my $j (0 .. $vocab_sz - 1) {
            if ($j == $flat_lab->[$i]) {
                $loss += -log($sm->[$i][$j]) / $loss_scale;
                $loss_grad->[$i][$j] = ($sm->[$i][$j] - 1) / $grad_scale;
            } else {
                $loss_grad->[$i][$j] = $sm->[$i][$j] / $grad_scale;
            }
        }
    }

    # Reshape gradient back to 3D
    my $gradient = [];
    for my $b (0 .. $BATCH_SIZE - 1) {
        for my $s (0 .. $seq_len - 1) {
            for my $v (0 .. $vocab_sz - 1) {
                $gradient->[$b][$s][$v] = $loss_grad->[$b * $seq_len + $s][$v];
            }
        }
    }

    $model->update(projection => 1, decode => 1, encode => 1,
                   gradient => $gradient, learning_rate => $LR);

    printf "Batch %d | loss %.4f\n", $batchno + 1, $loss;
}

# ── save final weights ────────────────────────────────────────────────────────
write_formatted_json($final_file, {
    src_embed        => $model->{src_embed}->get_weights(),
    tgt_embed        => $model->{tgt_embed}->get_weights(),
    encoder          => $model->{encoder}->get_weights(),
    decoder          => $model->{decoder}->get_weights(),
    projection_layer => $model->{projection_layer}->get_weights(),
});

say "Done. Diff $init_file vs $final_file on jsondiff.com";
