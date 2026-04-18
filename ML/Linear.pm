package ML::Linear;

use Modern::Perl;
use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use Math::Random::OO::Normal;
use ML::Util qw(print_2d_array print_1d_array add_2_arrays matmul linear transpose adam_optimiser clip_grad_norm);
use Storable qw(dclone);
use Carp qw(confess);
use Data::Dumper;

my $input_limit = 15000;

sub type {
   return "ML::Linear";
}

sub get_grad_tensors {
   my $self = shift;
   return [ grep { defined $_ } $self->{weights_grad}, $self->{bias_grad} ];
}

sub optimise {
   my $self = shift;
   my %args = @_;
   my $learning_rate = $args{learning_rate} || $self->{alpha};

   if ( $self->{debug} ) {
      print_2d_array( $self->name() . " bias gradient ",  $self->{bias_grad} );
      print_2d_array( $self->name() . " biases ",  $self->{biases} );
   }
   adam_optimiser(  $self->{bias_grad} , $self->{m_biases} ,  $self->{v_biases} ,  $self->{biases} ,
                            $learning_rate, $self->{beta1}, $self->{beta2}, $self->{epoch} );
   if ( $self->{debug} ) {
      print_2d_array( "bias afer update ", $self->{biases}  );
      print_2d_array( "m_bias afer update ", $self->{m_biases}  );
      print_2d_array( "v_bias afer update ", $self->{v_biases}  );
   }

   if ( $self->{debug} ) {
      print_2d_array($self->{name} .  "weights_grad " , $self->{weights_grad});
      print_2d_array($self->{name} .  "weights before " , $self->{W});
   }
   adam_optimiser(  $self->{weights_grad} ,  $self->{m_W} ,  $self->{v_W} ,  $self->{W} ,
                            $learning_rate, $self->{beta1}, $self->{beta2}, $self->{epoch} );
   if ( $self->{debug} ) {
      print_2d_array($self->{name} .  "W" , $self->{W});
      print_2d_array($self->{name} .  "m_W" , $self->{m_W});
      print_2d_array($self->{name} .  "v_W" , $self->{v_W});
   }
   $self->{epoch}++;
   $self->{bias_grad} = undef;
   $self->{weights_grad} = undef;
}


sub backward {
   my $self = shift;
   my %params = @_;
   my $next = $params{ next };
   my $in_grad = $next->gradient();
   my $is3d = 0;
   my $batch_size;
   if (ref($in_grad->[0][0]) eq "ARRAY") {
      $is3d = 1;
      $batch_size = scalar(@$in_grad);
   }
   say "is3d = $is3d" if $self->{debug};
   my $backward_defined_bias_grad = 1;
   if (!defined($self->{bias_grad})) {
      $backward_defined_bias_grad = 0;
      if ($is3d) {
         $self->{bias_grad} = dclone($in_grad->[0]);
         say "in_grad = " .Dumper($in_grad) if $self->{debug};
         if (defined($in_grad->[0][0]) and ref($in_grad->[0][0]) eq "ARRAY") {
            print_2d_array("in_grad 0", $in_grad->[0]) if $self->{debug};
         } else {
            print_2d_array("in_grad", $in_grad) if $self->{debug};
         }
         foreach my $b ( 1 .. $batch_size - 1) {
            print_2d_array("in_grad $b", $in_grad->[$b]) if $self->{debug};
            $self->{bias_grad} = add_2_arrays($self->{bias_grad}, $in_grad->[$b]);
            print_2d_array("bias_grad", $self->{bias_grad}) if $self->{debug};
         }
      } else {
         $self->{bias_grad} = dclone($in_grad);
      }
   } else {
      if ($is3d) {
         foreach my $b ( 0 .. $batch_size - 1) {
            print_2d_array("in_grad $b", $in_grad->[$b]) if $self->{debug};
            $self->{bias_grad} = add_2_arrays($self->{bias_grad}, $in_grad->[$b]);
            print_2d_array("bias_grad", $self->{bias_grad}) if $self->{debug};
         }
      } else {
         eval {
            my $tmp_in_grad = $in_grad;
            if (scalar(@{$in_grad->[0]}) == $self->{outsize}) {
               say "input gradient columns == outsize, transposing" if $self->{debug};
               $tmp_in_grad = transpose($in_grad);
            }
            my $tmp_in_grad_collapsed;
            if (scalar(@{$tmp_in_grad->[0]}) > 1) {
               say "input gradient columns > 1, need to collapse array" if $self->{debug};
               foreach my $i (0 .. $self->{outsize} - 1) {
                  $tmp_in_grad_collapsed->[$i][0] = 0;
                  foreach my $j (0 .. scalar(@{$tmp_in_grad->[0]}) - 1) {
do {
   print_2d_array("in_grad", $in_grad);
   print_2d_array("transposed in_grad" , $tmp_in_grad);
   die "tmp_in_grad $i $j undefined" 
} unless defined($tmp_in_grad->[$i][$j]);
                     $tmp_in_grad_collapsed->[$i][0] += $tmp_in_grad->[$i][$j];
                  }
               }
            } else {
               $tmp_in_grad_collapsed = $tmp_in_grad;
            }
            $self->{bias_grad} = add_2_arrays($self->{bias_grad}, $tmp_in_grad_collapsed);
         };
         if ($@) {
            say $@;
            print_2d_array("bias_grad", $self->{bias_grad});
            print_2d_array("in_grad", $in_grad);
            confess "add_2_arrays failed!";
         }         
      }
   }
   if ($is3d) {
      # bias needs to be 1 column wide, so add up all of the columns
      my $tmp_bias_grid = [];
      my $transposed_grads = transpose($self->{bias_grad});
      print_2d_array("transposed_grads", $transposed_grads) if $self->{debug};
      foreach my $x (0 .. scalar(@{$transposed_grads}) - 1) {
         $tmp_bias_grid->[$x][0] = 0;
         foreach my $y (0 .. scalar(@{$transposed_grads->[0]}) - 1) {
            $tmp_bias_grid->[$x][0] += $transposed_grads->[$x][$y];
         }
      }
      print_2d_array("tmp_bias_grid", $tmp_bias_grid) if $self->{debug};
      $self->{bias_grad} = $tmp_bias_grid;
      print_2d_array("bias_grad", $self->{bias_grad}) if $self->{debug};
   }
   my $ptr = $self->{bias_grad};
   my @bg_dims = ();
   while (ref($ptr) eq "ARRAY") {
      push @bg_dims, scalar(@$ptr);
      $ptr = $ptr->[0];
   }
   if (scalar(@bg_dims) == 2) {
      say "bias_grad dims == 2, OK!" if $self->{debug};
      say "bias grad dims = " . join(", ", @bg_dims) if $self->{debug};
      if ($bg_dims[0] != $self->{outsize}) {
         say "first bias_grad dimension == " . $bg_dims[0] . " expecting " . $self->{outsize} if $self->{debug};
         if ($bg_dims[1] != $self->{outsize}) {
            confess "I don't know what to do with a input gradient of " . join(" x ", @bg_dims) . " one of the dims should be the same as outsize! (" . $self->{outsize} . ")";
         }
         say "second dimension == outsize, transposing input gradient"  if $self->{debug};
         $self->{bias_grad} = transpose($self->{bias_grad});
      }
      if ($bg_dims[1] != 1) {
         if ($self->{debug}) {
            say "second bias_grad dimension == " . $bg_dims[1] . " expecting 1, collapsing input gradient to single column";
            say "is3d = $is3d" if $self->{debug}; 
            say "bias_grad previously defined = " . $backward_defined_bias_grad; 
            say "in_grad = " . print_2d_array("input gradient", $in_grad);
            say "outsize = " . $self->{outsize};
            say "insize = " . $self->{insize};
         }
         my $tmp_bias_grad = []; # will be outsize rows x 1 column
         foreach my $i (0 .. $self->{outsize} - 1) {
            $tmp_bias_grad->[$i][0] = 0;
            foreach my $j (0 .. scalar(@{$self->{bias_grad}->[0]}) - 1) {
               $tmp_bias_grad->[$i][0] += $self->{bias_grad}[$i][$j];
            }
         }
         $self->{bias_grad} = dclone($tmp_bias_grad); 
      }
   } else {
      die "bias_grad dims = " . join(", ", @bg_dims) . " expecting " . $self->{outsize} . ", 1";
   }

   if ($self->{debug}) {
      print_2d_array("Linear backward, bias_grad", $self->{bias_grad});
      if ($is3d) {
         foreach my $b ( 0 .. $batch_size - 1) {
            print_2d_array("Linear backward, transposed input $b", transpose($self->{input}[$b]));
         }
      } else {
         print_2d_array("Linear backward, transposed input", transpose($self->{input}));
      }
   }
   if (!defined($self->{weights_grad})) {
      if ($is3d) {
         if ($self->{debug}) {
            print_2d_array("ML::Linear self input 0", $self->{input}->[0]);
            print_2d_array("ML::Linear in_grad 0", $in_grad->[0]);
         }
         $self->{weights_grad} = matmul(transpose($self->{input}->[0]),$in_grad->[0] );
         foreach my $b ( 1 .. $batch_size - 1) {
            $self->{weights_grad} = add_2_arrays($self->{weights_grad} , matmul(transpose($self->{input}->[$b]), $in_grad->[$b]));
         }
      } else {
         $self->{weights_grad} = matmul(transpose($self->{input}),$in_grad );
      }
   } else {
      if ($is3d) {
         foreach my $b ( 0 .. $batch_size - 1) {
            $self->{weights_grad} = add_2_arrays($self->{weights_grad} , matmul(transpose($self->{input}->[$b]), $in_grad->[$b]));
         }
      } else {
         $self->{weights_grad} = add_2_arrays($self->{weights_grad} , matmul(transpose($self->{input}), $in_grad));
      }
   }
   print_2d_array("Linear backward, weights grad", $self->{weights_grad}) if $self->{debug};
   if ($is3d) {
      foreach my $b ( 0 .. $batch_size - 1) {
         $self->{gradient}[$b] =  matmul($in_grad->[$b],transpose($self->{W})); 
      }
   } else {
      $self->{gradient} =  matmul($in_grad,transpose($self->{W})); 
   }
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub initialise_weights {
   my $self = shift;
   $self->{biases} = [];
   $self->{m_biases} = [];
   $self->{v_biases} = [];

   foreach my $b (0 .. $self->{outsize} - 1) {
      $self->{biases}[$b][0] = 0;
      $self->{m_biases}[$b][0] = 0;
      $self->{v_biases}[$b][0] = 0;
   }

   $self->{W} = [];
   my $prng = Math::Random::OO::Normal->new(0, sqrt(2 /  $self->{insize}));
   foreach my $row ( 0 .. $self->{insize} - 1) {
      foreach my $col ( 0 .. $self->{outsize} - 1) {
         $self->{W}[$row][$col] = $prng->next();
         $self->{m_W}[$row][$col] = 0;
         $self->{v_W}[$row][$col] = 0;
      }
   }
}

sub set_weights_and_biases {
   my $self = shift;
# we expect to see input_channels x output_channels weights arrays, with each array containing kernel_size^2 entries
   my %args = @_;
   if (!defined($args{weights})) { 
      # randomly initialise weights
      #say "initialise weights and biases randomly";
      $self->initialise_weights();
   } else {
      $self->{W} = $args{weights}; 
      $self->{biases} = $args{biases};
      foreach my $b (0 .. $self->{outsize} - 1) {
         $self->{m_biases}[$b][0] = 0;
         $self->{v_biases}[$b][0] = 0;
      }
      foreach my $row ( 0 .. $self->{insize} - 1) {
         foreach my $col ( 0 .. $self->{outsize} - 1) {
            $self->{m_W}[$row][$col] = 0;
            $self->{v_W}[$row][$col] = 0;
         }
      }
   }
   $self->{bias_grad} = undef;
   $self->{weight_grad} = undef;
}


sub forward {
# this is optimised for Transformer models, so may not work with CNN
   my $self = shift;
   my %args = @_;
   $self->{alpha} = 0.1;
   $self->{beta1} = 0.9;
   $self->{beta2} = 0.98;#99;

   my $batch = $args{batch};
if (ref($batch) eq "") {
   die "ML::Linear forward input batch not defined, called by " . join(", ", caller());
}
   $self->{input} = dclone($batch);
   my $N = scalar(@{$batch}); # batch size
   $self->{batch_size} = $N;
   my $is3d = 0;
   if (ref($self->{input}[0][0]) eq "ARRAY") {
      $is3d = 1;
   }
   # assume image is square
   my $side = $self->{outsize};
   if ($self->{debug}) {
      print_2d_array("W", $self->{W});
      if ($is3d) {  
         foreach my $b (0 .. $N - 1) {
            print_2d_array("batch $b", $batch->[$b]);
         }
      } else {
         print_2d_array("batch", $batch);
      }
      print_2d_array("biases", $self->{biases});
   }
if (!defined($self->{W})) {
   die "ML::Linear W not defined, created by " . $self->{created_by};
}
if (!defined($self->{W}[0])) {
   die "ML::Linear W[0] not defined, created by " . $self->{created_by};
}
   if ($self->{debug}) {
      say "size of W = " . scalar(@{$self->{W}}) . " x " . scalar(@{$self->{W}[0]});
      if ($is3d) {
         say "size of each batch = " . scalar(@{$batch->[0]}) . " x " . scalar(@{$batch->[0][0]}) ;
      } else {
         say "size of input = " . scalar(@$batch) . " x " . scalar(@{$batch->[0]});
      }
      print_2d_array("biases", $self->{biases});
   }
   my $output = [];
   if ($is3d) {
      foreach my $b (0 .. $N - 1) {
         $output->[$b] = linear($batch->[$b], $self->{W}, $self->{biases});
      }
   } else {
      $output = linear($batch, $self->{W}, $self->{biases});
   }
   return $output;
}
 
sub outsize {
   my $self = shift;
   return $self->{outsize};
}

sub name {
   my $self = shift;
   return $self->{name};
}

sub get_weights {
   my $self = shift;
   return {
           insize => $self->{insize},
           outsize => $self->{outsize},
           weights => $self->{W},
           biases => $self->{biases}
          };
}

sub new {
   my $class = shift;
   my $self = {};
   my %args = @_;
   if (defined($args{debug}) and $args{debug} == 1) {
      $self->{debug} = 1;
   } else {
      $self->{debug} = 0;
   }
   $self->{name} = $args{name} || "Linear";
   my $debug = $args{debug};
   if (defined($debug) and $debug =~ /^\d+$/ and $debug > 0) { 
      $debug = 1;
   } else {
      $debug = 0;
   }
   $self->{debug} = $debug;
   $self->{epoch} = 1;
   my $input_size = $args{insize};
   my $outsize = $args{outsize};
   if (!defined($input_size) or $input_size !~ /^\d+$/ or $input_size > $input_limit) {
      die "Invalid insize parameter '$input_size', must be integer <= $input_limit";
   }
   if (!defined($outsize) or $outsize !~ /^\d+$/) {
      confess "Invalid outsize parameter '$outsize', must be integer ";
   }

   $self->{insize} = $input_size;
   $self->{outsize} = $args{outsize};
   $self->{created_by} = join(", ", caller());
   say "ML::Linear outsize = " . $self->{outsize}  if $self->{debug};
   return bless $self, $class;
}  

1;
