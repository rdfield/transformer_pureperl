package ML::Encoder;

use Modern::Perl;
use ML::LayerNormalisation;
use ML::EncoderBlock;
use ML::MHA;
use ML::PositionwiseFeedForward;

use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);

sub type {
   return "ML::Encoder";
}

sub initialise {
   my $self = shift;
   $self->{ln_layer}->initialise();
}

sub get_grad_tensors {
   my $self = shift;
   return [ (map { @{$_->get_grad_tensors()} } @{$self->{EncoderBlocks}}),
            @{$self->{norm}->get_grad_tensors()} ];
}

sub optimise {
   my $self = shift;
   foreach my $l (0 .. $self->{layers} - 1) {
      $self->{EncoderBlocks}[$l]->optimise( @_ );
   }
   $self->{norm}->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   $self->{input} = dclone($args{batch});

   my $output = $self->{input};
   $self->{mini_batch_size} = scalar(@{$self->{input}});
   foreach my $l (0 .. $self->{layers} - 1) {
      $output = $self->{EncoderBlocks}[$l]->forward( batch => $output, mask => $args{mask} );
   }
   $output = $self->{norm}->forward( batch => $output );

   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{mini_batch_size} ||= $params{mini_batch_size};
 
   my $next = $params{ next };
   #my $input_gradient = dclone($next->gradient());
   $self->{norm}->backward( next => $next );
   my $local_next = $self->{norm};
   if ($self->{debug}) {
      foreach my $b ( 0 .. $self->{mini_batch_size} - 1) {
         print_2d_array("Encoder backward grad after norm", $self->{norm}->{gradient}[$b]);
      }
   }
   foreach my $l (reverse @{$self->{EncoderBlocks}}) {
       $l->backward( next => $local_next );
       $local_next = $l;
   }
   $self->{gradient} = $self->{EncoderBlocks}[0]->gradient();
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
   #return $self->{bn}->gradient(@_);
}

sub outsize {
   my $self = shift;
   return $self->{insize};
}

sub name {
   my $self = shift;
   return $self->{name};
}

sub get_weights {
   my $self = shift;
   my $weights = { embeddings => $self->{embeddings},
                   max_seq_length => $self->{max_seq_length},
                   layers => $self->{layers},
                   num_heads => $self->{num_heads},
                   d_ff => $self->{d_ff},
                   dropout => $self->{dropout},
                   normalisation => $self->{norm}->get_weights(),
                 };
   foreach my $i (0 .. $self->{layers} - 1) {
      push @{$weights->{encoderblocks}}, $self->{EncoderBlocks}[$i]->get_weights();
   }
   return $weights;
}

sub new {
   my $class = shift;
   my %args = @_;

   my $embeddings = $args{embeddings};
   die "Encoder invalid embeddings: " . $args{embeddings} unless defined($embeddings) and $embeddings =~ /^\d+$/;
   my $max_seq_length = $args{max_seq_length};
   die "Encoder invalid max_seq_length: " . $args{max_seq_length} unless defined($max_seq_length) and $max_seq_length =~ /^\d+$/;
   my $layers = $args{layers};
   die "Encoder invalid layers: " . $args{layers} unless defined($layers) and $layers =~ /^\d+$/;
   my $num_heads = $args{num_heads};
   die "Encoder invalid num_heads: " . $args{num_heads} unless defined($num_heads) and $num_heads =~ /^\d+$/;
   my $d_ff = $args{d_ff};
   die "Encoder invalid d_ff: " . $args{d_ff} unless defined($d_ff) and $d_ff =~ /^\d+$/;
   my $dropout = $args{dropout};
   die "Encoder invalid dropout: " . $args{dropout} unless defined($dropout) and $dropout =~ /^\d+\.\d+$/;
   my $self = {};
   $self->{embeddings} = $embeddings;
   $self->{max_seq_length} = $max_seq_length;
   $self->{layers} = $args{layers};
   $self->{num_heads} = $args{num_heads};
   $self->{d_ff} = $args{d_ff};
   $self->{dropout} = $args{dropout};
   $self->{EncoderBlocks} = [];
   foreach my $i (0 .. $self->{layers} - 1) {
      if (defined($args{encoderblocks}->[$i])) {
         push @{$self->{EncoderBlocks}}, ML::EncoderBlock->new( %{ $args{encoderblocks}->[$i] } );
      } else {
         push @{$self->{EncoderBlocks}}, ML::EncoderBlock->new( embeddings => $self->{embeddings}, max_seq_length => $self->{max_seq_length}, num_heads => $self->{num_heads}, d_ff => $self->{d_ff}, dropout => $self->{dropout});
      }
   }
   if (defined($args{normalisation})) {
      $self->{norm} = ML::LayerNormalisation->new( embeddings => $args{normalisation}->{embeddings}, max_seq_length => $args{normalisation}->{max_seq_length});
      $self->{norm}->initialise( weights => { gamma => $args{normalisation}->{gamma}, beta => $args{normalisation}->{beta} } );
   } else {
      $self->{norm} = ML::LayerNormalisation->new( embeddings => $self->{embeddings}, max_seq_length => $self->{max_seq_length});
      $self->{norm}->initialise( );
   }
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "encoder";
   return bless $self, $class;
}  

1;
