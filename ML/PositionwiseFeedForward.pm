package ML::PositionwiseFeedForward;

use Modern::Perl;
use ML::Linear;
use ML::LinearReLu;
use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);

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
           activation => $self->{activation}->{get_weights},
           dropout => $self->{dropout},
           l2 => $self->{l2}->get_weights(),
           embeddings =>  $self->{insize},
           d_ff => $self->{dffsize},
          };
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

   my $output = [];
   foreach my $b (0 .. scalar(@{$self->{input}}) - 1) {
      if ($self->{debug}) {
         print_2d_array("pff input $b", $self->{input}[$b]);
      }
      $output->[$b] = $self->{l1}->forward( batch => $self->{input}[$b] );
      $output->[$b] = $self->{activation}->forward( batch => $output->[$b] );
      if (defined($self->{dropout})) {
         foreach my $x (0 .. $#{$output->[0]}) {
            foreach my $y (0 .. $#{$output->[0][0]}) {
               my $rand = rand();
               if ($rand < $self->{dropout}) {
                  $output->[$b][$x][$y] = 0;
               }
            }
         }
      }
      $output->[$b] = $self->{l2}->forward( batch => $output->[$b] );
   }

   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{batch_size} ||= $params{batch_size};
 
   my $next = $params{ next };
   if ($self->{debug}) {
      say "pff backward next => " . ref($next);
   }

   my $in_gradients = $next->gradient();

   foreach my $b ( 0 .. $self->{batch_size} - 1) {
      {
        package ML::Dummy;
        sub new { my $class = shift; bless { @_ }, $class }
        sub gradient { shift->{data} }
      }
      if ($self->{debug}) {
         print_2d_array("pff backward input gradients $b", $in_gradients->[$b]);
      }
      my $dummy_gradients = ML::Dummy->new(data => $in_gradients->[$b]);

      $self->{l2}->backward( next => $dummy_gradients );
      $self->{activation}->backward( next => $self->{l2} );
      $self->{l1}->backward( next => $self->{activation} );
      if ($self->{debug}) {
         print_2d_array("activation backward gradients", $self->{activation}->gradient());
         print_2d_array("l2 backward gradients", $self->{l2}->gradient());
         print_2d_array("l1 backward gradients", $self->{l1}->gradient());
      }
      $self->{gradient}->[$b] = $self->{l1}->gradient();
   }
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
