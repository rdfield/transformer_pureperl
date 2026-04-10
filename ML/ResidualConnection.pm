package ML::ResidualConnection;

use Modern::Perl;
use ML::Linear;
use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);
use Carp qw(confess cluck);

sub type {
   return "ML::ResidualConnection";
}

sub initialise {
   my $self = shift;
   $self->{ln_layer}->initialise();
}

sub optimise {
   my $self = shift;
   $self->{sublayer}->optimise( @_ );
   $self->{ln_layer}->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   $self->{input} = dclone($args{batch});

   if ($self->{debug}) {
      cluck "ML::ResidualConnection forward called";
   }

   my $output = [];
   if ($self->{debug}) {
      say "input dimentions = " . scalar(@{$self->{input}}) . ", " . scalar(@{$self->{input}[0]}) . ", " . scalar(@{$self->{input}[0][0]});
   }
   if (ref($args{sublayer}) eq "ML::Linear") { # expects 2d
      foreach my $b ( 0 .. scalar(@{$self->{input}}) - 1) {
         $output->[$b] = $args{sublayer}->forward( batch => $self->{input}->[$b]);
      }
      $self->{batch_size} = scalar(@{$self->{input}});
   } else {
      $output = $args{sublayer}->forward( batch => $self->{input}, batch_1 => $args{batch_1}, batch_2 => $args{batch_2}, mask => $args{mask} );
   }
   $self->{sublayer} = $args{sublayer};
   foreach my $b (0 .. scalar(@{$self->{input}}) - 1) {
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
   }
   foreach my $b ( 0 .. scalar(@{$self->{input}}) - 1) {
      $output->[$b] = add_2_arrays($output->[$b], $self->{input}->[$b]);
   }
   $output = $self->{ln_layer}->forward( batch => $output );
   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{batch_size} ||= $params{batch_size};
 
   my $next = $params{ next };
   if ($self->{debug}) {
      say "residual_connection backward next => " . ref($next);
   }
   my $in_gradient = dclone($next->gradient());
   $self->{ln_layer}->backward( next => $next );
   if ($self->{debug}) {
      my $ln_gradients = $self->{ln_layer}->gradient();
      if ($self->{debug}) {
         foreach my $b (0 .. $self->{batch_size} - 1) {
            print_2d_array("ln_layer gradient $b", $ln_gradients->[$b]);
         }
      }
   }
   my $ln_gradients = $self->{ln_layer}->gradient();
   if ($self->{debug}) {
      say "residual connection sublayer => " . ref($self->{sublayer});
   }
   $self->{sublayer}->backward(next => $self->{ln_layer});
   $self->{gradient} = $self->{sublayer}->gradient();
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub gradient_kv {
   my $self = shift;
   return $self->{sublayer}->can('gradient_kv') ? $self->{sublayer}->gradient_kv() : undef;
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
   return {
       embeddings => $self->{embeddings},
       max_seq_length => $self->{max_seq_length},
       ln_layer => $self->{ln_layer}->get_weights(),
       dropout => $self->{dropout}
   }
}

sub new {
   my $class = shift;
   my %args = @_;

   my $embeddings = $args{embeddings};
   die "ResidualConnection invalid embeddings: " . $args{embeddings} unless defined($embeddings) and $embeddings =~ /^\d+$/;
   my $max_seq_length = $args{max_seq_length};
   die "ResidualConnection invalid max_seq_length: " . $args{max_seq_length} unless defined($max_seq_length) and $max_seq_length =~ /^\d+$/;
   my $dropout = $args{dropout};
   my $self = {};
   $self->{embeddings} = $embeddings;
   $self->{max_seq_length} = $max_seq_length;
   $self->{created_by} =  "Residual connection created with max_seq_length $max_seq_length by " . join(", ", caller());
   $self->{ln_layer} = ML::LayerNormalisation->new(embeddings => $self->{embeddings}, max_seq_length => $self->{max_seq_length}, creator => $self->{created_by} . " " . $args{creator});
   $self->{ln_layer}->initialise();
   $self->{dropout} = $dropout;
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "residualconnection";
   return bless $self, $class;
}  

1;
