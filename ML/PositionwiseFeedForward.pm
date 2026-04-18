package ML::PositionwiseFeedForward;

use Modern::Perl;
use ML::Linear;
use ML::LinearReLu;
use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);
use ML::LossGradient;

sub type {
   return "ML::PositionwiseFeedForward";
}

sub initialise {
   my $self = shift;
   $self->{l1}->set_weights_and_biases();
   $self->{l2}->set_weights_and_biases();
   $self->{activation}->initialise();
}

sub get_weights {
   my $self = shift;
   return { 
           l1 => $self->{l1}->get_weights(),
           dropout => $self->{dropout},
           l2 => $self->{l2}->get_weights(),
           embeddings =>  $self->{insize},
           d_ff => $self->{dffsize},
          };
}
   
       
sub get_grad_tensors {
   my $self = shift;
   return [ @{$self->{l1}->get_grad_tensors()}, @{$self->{l2}->get_grad_tensors()} ];
}

sub optimise {
   my $self = shift;
   $self->{l1}->optimise( @_ );
   $self->{l2}->optimise( @_ );
   $self->{activation}->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   $self->{input} = dclone($args{batch});
   $self->{batch_size} = scalar(@{$self->{input}});

   # Pass the full 3D batch through l1, activation, l2 so that each layer
   # stores all batch slices in its {input} — required for correct backward.
   my $output = $self->{l1}->forward( batch => $self->{input} );
   $output = $self->{activation}->forward( batch => $output );
   $self->{dropout_mask} = undef;
   if (defined($self->{dropout})) {
      my $scale = 1.0 / (1.0 - $self->{dropout});
      for my $b (0 .. $self->{batch_size} - 1) {
         for my $x (0 .. $#{$output->[$b]}) {
            for my $y (0 .. $#{$output->[$b][$x]}) {
               if (rand() < $self->{dropout}) {
                  $self->{dropout_mask}[$b][$x][$y] = 0;
                  $output->[$b][$x][$y] = 0;
               } else {
                  $self->{dropout_mask}[$b][$x][$y] = $scale;
                  $output->[$b][$x][$y] *= $scale;
               }
            }
         }
      }
   }
   $output = $self->{l2}->forward( batch => $output );
   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;

   $self->{batch_size} ||= $params{batch_size};

   my $next = $params{next};
   if ($self->{debug}) {
      say "pff backward next => " . ref($next);
   }

   # Backprop through l2 → activation → l1 using the full 3D gradient.
   $self->{l2}->backward( next => $next );
   # Apply dropout mask to the gradient flowing from l2 back into activation.
   if (defined($self->{dropout}) && defined($self->{dropout_mask})) {
      my $masked = dclone($self->{l2}->gradient());
      for my $b (0 .. $#$masked) {
         for my $x (0 .. $#{$masked->[$b]}) {
            for my $y (0 .. $#{$masked->[$b][$x]}) {
               $masked->[$b][$x][$y] *= $self->{dropout_mask}[$b][$x][$y];
            }
         }
      }
      $self->{activation}->backward( next => ML::LossGradient->new(gradient => $masked) );
   } else {
      $self->{activation}->backward( next => $self->{l2} );
   }
   $self->{l1}->backward( next => $self->{activation} );
   $self->{gradient} = $self->{l1}->gradient();
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub outsize {
   my $self = shift;
   return $self->{insize};
}

sub name {
   my $self = shift;
   return $self->{name};
}

sub new {
   my $class = shift;
   my %args = @_;
   my $insize = $args{embeddings}; 
#say Dumper(\%args);
   die "PositionwiseFeedForward invalid embeddings: " . $insize unless defined($insize) and $insize =~ /^\d+$/;
   my $dffsize = $args{d_ff};
   die "PositionwiseFeedForward invalid d_ff: " . $dffsize unless defined($dffsize) and $dffsize =~ /^\d+$/;
   my $outsize = $args{insize}; # is just insize, i.e. number of embeddings
   my $dropout = $args{dropout};
   my $self = {};
   $self->{l1} = ML::Linear->new(insize => $insize, outsize => $dffsize, debug => $self->{debug});
   $self->{l1}->set_weights_and_biases( weights => $args{l1}->{weights} , biases => $args{l1}->{biases} );
   $self->{activation} = ML::LinearReLu->new(insize => $dffsize);
   $self->{dropout} = $dropout;
   $self->{l2} = ML::Linear->new(insize => $dffsize, outsize => $insize, debug => $self->{debug});
   $self->{l2}->set_weights_and_biases( weights => $args{l2}->{weights} , biases => $args{l2}->{biases} );
   $self->{insize} = $insize;
   $self->{dffsize} = $dffsize;
   $self->{outsize} = $insize;
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "positionwisefeedforward";
   return bless $self, $class;
}  

1;
