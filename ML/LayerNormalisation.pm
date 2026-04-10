package ML::LayerNormalisation;
use Modern::Perl;
use ML::Util qw(print_2d_array adam_optimiser print_1d_array);
use Data::Dumper;
use Storable qw(dclone);
use Carp;

my $channel_limit = 1500;

sub type {
   return "ML::LayerNormalisation";
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub get_weights {
   my $self = shift;
   return { gamma => $self->{gamma},
            beta => $self->{beta},
            max_seq_length => $self->{max_seq_length},
            embeddings => $self->{embeddings}
          };
}

sub set_weights {
   my $self = shift;
   my $args = shift;
   $self->{gamma} = dclone( $args->{gamma} );
   $self->{beta} = dclone( $args->{beta} );
}
   
sub optimise {
   my $self = shift;
   my %args = @_;
   my $lr    = $args{learning_rate} || 0.001;
   my $beta1 = 0.9;
   my $beta2 = 0.999;
   my $eps   = 1e-8;
   my $t     = $self->{adam_epoch};
   foreach my $e (0 .. $self->{embeddings} - 1) {
      # Adam update for gamma
      $self->{m_gamma}[$e] = $beta1 * $self->{m_gamma}[$e] + (1 - $beta1) * $self->{dgamma}[$e];
      $self->{v_gamma}[$e] = $beta2 * $self->{v_gamma}[$e] + (1 - $beta2) * $self->{dgamma}[$e] ** 2;
      my $m_hat = $self->{m_gamma}[$e] / (1 - $beta1 ** $t);
      my $v_hat = $self->{v_gamma}[$e] / (1 - $beta2 ** $t);
      $self->{gamma}[$e] -= $lr * $m_hat / (sqrt($v_hat) + $eps);
      $self->{dgamma}[$e] = 0;
      # Adam update for beta
      $self->{m_beta}[$e] = $beta1 * $self->{m_beta}[$e] + (1 - $beta1) * $self->{dbeta}[$e];
      $self->{v_beta}[$e] = $beta2 * $self->{v_beta}[$e] + (1 - $beta2) * $self->{dbeta}[$e] ** 2;
      $m_hat = $self->{m_beta}[$e] / (1 - $beta1 ** $t);
      $v_hat = $self->{v_beta}[$e] / (1 - $beta2 ** $t);
      $self->{beta}[$e] -= $lr * $m_hat / (sqrt($v_hat) + $eps);
      $self->{dbeta}[$e] = 0;
   }
   $self->{adam_epoch}++;
   if ( $self->{debug} ) {
      say "gamma after update " . join(",", @{$self->{gamma}});
      say "beta after update "  . join(",", @{$self->{beta}});
   }
}

sub backward {
   my $self = shift;
   my %params = @_;
   $self->{batch_size} ||= $params{batch_size};
   my $next = $params{ next };
   my $delta = $next->gradient();
   if ($self->{debug} == 1) {
      foreach my $b (0 .. $self->{batch_size} - 1) {
         print_2d_array("layernorm back prop delta $b", $delta->[$b]) if $self->{debug};
      }
   }
   $self->{dgamma} = [];
   foreach my $e (0 .. $self->{embeddings} - 1) {
      $self->{dgamma}[$e] = 0;
      $self->{dbeta}[$e] = 0;
   }
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $self->{dgamma}[$e] += $delta->[$b][$s][$e] * $self->{mu}[$b][$s][$e] * ( 1 / $self->{sigma}[$b][$s] );
            $self->{dbeta}[$e] += $delta->[$b][$s][$e];
         }
      }
   }
   if ($self->{debug}) {
      say "dgamma = " . join(", ", @{$self->{dgamma}});
      say "dbeta = " . join(", ", @{$self->{dbeta}});
   }
   $self->{gradient} = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         # pre-compute the two sums over the embedding dimension needed for the
         # correct LayerNorm gradient:
         #   dx_i = (gamma_i/sigma) * (delta_i - mean_j(delta_j)
         #                                      - x_hat_i * mean_j(delta_j * x_hat_j))
         # Correct LayerNorm input gradient:
         #   dL/dX[k] = (1/σ) * [ δ[k]·γ[k]
         #                        − mean_e(δ[e]·γ[e])
         #                        − x̂[k] · mean_e(δ[e]·γ[e]·x̂[e]) ]
         # gamma must be INSIDE both means, not factored outside.
         my $sum_delta_gamma      = 0;
         my $sum_delta_gamma_xhat = 0;

         foreach my $e (0 .. $self->{embeddings} - 1) {
            my $x_hat = $self->{mu}[$b][$s][$e] / $self->{sigma}[$b][$s];
            $sum_delta_gamma      += $delta->[$b][$s][$e] * $self->{gamma}[$e];
            $sum_delta_gamma_xhat += $delta->[$b][$s][$e] * $self->{gamma}[$e] * $x_hat;
         }
         my $mean_delta_gamma      = $sum_delta_gamma      / $self->{embeddings};
         my $mean_delta_gamma_xhat = $sum_delta_gamma_xhat / $self->{embeddings};
         foreach my $e (0 .. $self->{embeddings} - 1) {
            my $x_hat = $self->{mu}[$b][$s][$e] / $self->{sigma}[$b][$s];
            $self->{gradient}[$b][$s][$e] = (1 / $self->{sigma}[$b][$s])
                     * (  $delta->[$b][$s][$e] * $self->{gamma}[$e]
                        - $mean_delta_gamma
                        - $x_hat * $mean_delta_gamma_xhat );
         }
      }
      if ($self->{debug}) {
         print_2d_array("LayerNorm gradient $b", $self->{gradient}[$b]) if $self->{debug};
      }
   }
}

sub new {
   my $class = shift;
   my $self = {};
   my %config = @_;
   $self->{max_seq_length} = $config{max_seq_length}; # for memory reservation
   $self->{embeddings} = $config{embeddings};
   $self->{name} = $config{name} || "layernormalisation";
   #$self->{outsize} = $config{seq_length} * $config{embeddings};
   $self->{debug} = $config{debug};
   $self->{debug} ||= 0;
   $self->{out} = [];
   $self->{created_by} = join(", ", caller());
   $self->{creator} = $config{creator};
   return bless $self, $class;
}

sub outsize {
  my $self = shift;
  return $self->{max_seq_length};
}

sub name {
  my $self = shift;
  return $self->{name};
}

sub initialise {
   my $self = shift;
   # parameters will be $self->{embeddings} wide
   my %args = @_;
   $self->{eps} = 0.00001;
   if (defined($args{weights})) {
      $self->{gamma} = dclone($args{weights}->{gamma});
      $self->{beta}  = dclone($args{weights}->{beta});
      foreach my $c (0 .. $self->{embeddings} - 1) {
         $self->{m_gamma}[$c] = 0;
         $self->{m_beta}[$c]  = 0;
         $self->{v_gamma}[$c] = 0;
         $self->{v_beta}[$c]  = 0;
      }
      $self->{adam_epoch} = 1;
      return;
   }
   $self->{gamma} = [];
   $self->{beta}  = [];
   foreach my $c (0 .. $self->{embeddings} - 1) {
      $self->{gamma}[$c]   = 1;
      $self->{beta}[$c]    = 0;
      $self->{m_gamma}[$c] = 0;
      $self->{m_beta}[$c]  = 0;
      $self->{v_gamma}[$c] = 0;
      $self->{v_beta}[$c]  = 0;
   }
   $self->{adam_epoch} = 1;
}

sub forward {
   my $self = shift;
   my %args = @_;
   $self->{epoch} = $args{epoch} || 1;
   $self->{alpha} = 0.1;
   $self->{beta1} = 0.9;
   $self->{beta2} = 0.999;
   my $X = $args{batch};
   if ($self->{debug} == 1) {
      say "LayerNormalisation input shape: " . join(", ", scalar(@{$X}), scalar(@{$X->[0]}), scalar(@{$X->[0][0]}));
      say "layernormalisation forward input";
      say Dumper($X);
   }
   my $N = scalar(@{$X}); # batch size
   $self->{batch_size} = $N;
   $self->{seq_length} = scalar(@{$X->[0]});
   unless (scalar(@{$X->[0][0]}) == $self->{embeddings}) {
      confess "X input must have last dimension of " . $self->{embeddings} . " found " . scalar(@{$X->[0][0]});
   }
   $self->{m} = [];
   $self->{mu} = [];
   $self->{var} = [];
   $self->{sigma} = [];
# Step 1: Calculate mean per seq/batch
   foreach my $b ( 0 ..  $N - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         $self->{m}[$b][$s] = 0;
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $self->{m}[$b][$s] += $X->[ $b ][ $s ][ $e ] / $self->{embeddings}; 
         }
      }
   }
   print_2d_array("LayerNormalisation m" , $self->{m}) if $self->{debug};
# step 2 mu
   foreach my $b ( 0 ..  $N - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $self->{mu}[$b][$s][$e] = $X->[$b][$s][$e] - $self->{m}[$b][$s];
         }
      }
      print_2d_array("mu $b", $self->{mu}[$b]) if $self->{debug};
   }
# step 3 var & sigma
   foreach my $b ( 0 ..  $N - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         $self->{var}[$b][$s] = 0 ;
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $self->{var}[$b][$s] += $self->{mu}[$b][$s][$e] ** 2 / $self->{embeddings};
         }
         $self->{sigma}[$b][$s] = sqrt($self->{var}[$b][$s] + $self->{eps});
         $self->{std_inv}[$b][$s] = 1 / $self->{sigma}[$b][$s];
      }
   }
   print_2d_array("batch/seq variance",$self->{var}) if $self->{debug};
   print_2d_array("sigma",$self->{sigma}) if $self->{debug};
   print_2d_array("std_inv",$self->{std_inv}) if $self->{debug};

   $self->{out} = [];
   foreach my $b ( 0 ..  $N - 1) {
      foreach my $s (0 .. $self->{seq_length} - 1) {
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $self->{out}->[ $b ][ $s ][ $e ] = $self->{mu}->[ $b ][ $s ][ $e ] / $self->{sigma}[$b][$s] * $self->{gamma}[$e] + $self->{beta}->[ $e ]; 
         }
      } 
   }
   return $self->{out};
}

1;
