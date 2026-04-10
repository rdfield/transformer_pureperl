package ML::DataLoader;
use List::Util qw(shuffle);
use Storable qw(dclone);
use ML::Util qw(tokenize_sequence pad_sequence);

sub new {
   my $class = shift;
   bless { @_ }, $class
}

sub __len__ {
   my $self = shift;
   return scalar(@{$self->{data}});
}

sub batch_size {
   my $self = shift;
   return $self->{batch_size};
}

sub __getitem__ {
   my $self = shift;
   my $idx = shift;
   if ($idx eq "idx") {
      $idx = shift;
   }
   my $item = $self->{data}[$idx];

   my $src_ids = tokenize_sequence( sequence => $item->{src}, token_to_id_map => $self->{token_to_id} );
   my $src_padded = pad_sequence( sequence_ids => $src_ids, max_len => $self->{max_len}, pad_id => $self->{pad_id} );
   my $tgt_ids = tokenize_sequence(sequence => $item->{tgt}, token_to_id_map => $self->{token_to_id} );

   my $decoder_input_ids = dclone($tgt_ids);
   pop @$decoder_input_ids; # Exclude EOS
   my $decoder_input_padded = pad_sequence(sequence_ids => $decoder_input_ids, max_len => $self->{max_len}, pad_id => $self->{pad_id} );
   my $label_ids = dclone($tgt_ids);
   shift @$label_ids; # Exclude SOS
   my $label_padded = pad_sequence(sequence_ids => $label_ids, max_len => $self->{max_len}, pad_id => $self->{pad_id}); # Pad labels too

   return {
      "src_ids" =>  $src_padded,
      "decoder_input_ids" =>  $decoder_input_padded,
      "label_ids" =>  $label_padded
   };
}

sub init_iter {
   my $self = shift;
   my %args = @_;
   $self->{curbatch} = 0;
   $self->{curpos} = 0;
   $self->{batch_size} = $args{ batch_size };
   $self->{batches} = int( $self->__len__  / $self->{batch_size} );
   @{$self->{data}} = shuffle( @{$self->{data}} );
}

sub next_batch {
   my $self = shift;
   my $batch = {};
   foreach my $i (0 .. $self->{batch_size} - 1) {
      my $item = $self->__getitem__( idx => $self->{curpos}++ );
      push @{$batch->{src_ids}}, $item->{src_ids};
      push @{$batch->{decoder_input_ids}}, $item->{decoder_input_ids};
      push @{$batch->{label_ids}}, $item->{label_ids};
   }
   $self->{curbatch}++;
   return $batch;
}

1;
