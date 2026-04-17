package ML::Embedding;

use lib '/home/mvine/imagenet';
use Modern::Perl;
use Math::Random qw(random_uniform);
use Math::Random::OO::Normal;
use ML::Util qw(print_2d_array print_1d_array add_2_arrays matmul linear transpose adam_optimiser clip_grad_norm);
use Storable qw(dclone);
use Data::Dumper;
use Carp qw(cluck);

my $input_limit = 1500;

sub type {
   return "ML::Embedding";
}

sub optimise {
   my $self = shift;
   my %args = @_;
   my $lr    = $args{learning_rate} // 0.001;
   my $beta1 = 0.9;
   my $beta2 = 0.999;
   my $t     = $self->{adam_step};
   my %updated;
   foreach my $b ( 0 .. $self->{batch_size} - 1) {
      foreach my $i (0 .. $self->{seq_length} - 1) {
         my $tok = $self->{input}[$b][$i];
         next if defined($self->{pad_id}) && $tok == $self->{pad_id};
         next if $updated{$tok}++;   # apply dW[tok] exactly once per step
         clip_grad_norm($self->{dW}[$tok]);
         foreach my $j (0 .. $self->{embeddings} - 1) {
            $self->{m_W}[$tok][$j] = $beta1 * $self->{m_W}[$tok][$j] + (1 - $beta1) * $self->{dW}[$tok][$j];
            $self->{v_W}[$tok][$j] = $beta2 * $self->{v_W}[$tok][$j] + (1 - $beta2) * $self->{dW}[$tok][$j] ** 2;
            my $m_hat = $self->{m_W}[$tok][$j] / (1 - $beta1 ** $t);
            my $v_hat = $self->{v_W}[$tok][$j] / (1 - $beta2 ** $t);
            $self->{W}[$tok][$j] -= $lr * $m_hat / (sqrt($v_hat) + 1e-8);
         }
      }
   }
   $self->{adam_step}++;
}


sub backward {
   my $self = shift;
   my %params = @_;
   my $next = $params{ next };
   my $in_grad = $next->gradient();
   $self->{weights_grad} = dclone($in_grad);
   if ($self->{debug}) {
      foreach my $b ( 0 .. $self->{batch_size} - 1) {
         print_2d_array("Embedding backward, weights grad batch $b", $self->{weights_grad}[$b]);
      }
   }
   # this will be a 3-d array - batch_size x seq_length x embeddings
   #                                   embeddings sometimes known as d_model
   # the delta will be vocab_size x embeddings
   $self->{dW} = [];
   foreach my $i ( 0 .. $self->{vocab_size} - 1) {
      foreach my $j ( 0 .. $self->{embeddings} - 1) {
         $self->{dW}->[$i][$j] = 0;
      }
   }
   foreach my $b ( 0 .. $self->{batch_size} - 1) {
      foreach my $s ( 0 .. $self->{seq_length} - 1) {
         foreach my $e ( 0 .. $self->{embeddings} - 1) {
            $self->{dW}->[$self->{input}[$b][$s]][$e] += $in_grad->[$b][$s][$e];
         }
      }
   }
   if ($self->{debug}) {
      print_2d_array("dW", $self->{dW});
   }
}

sub gradient {
   my $self = shift;
   $self->{gradient} = $self->{weights_grad}; # TODO not sure how to generate gradients through the embedding layer yet
   return $self->{gradient};
}

sub parameter_count {
   my $self = shift;
   return [ $self->{vocab_size}, $self->{embeddings} ]; 
}

sub initialise_weights {
   my $self = shift;
   $self->{W} = [];
   $self->{m_W} = [];
   $self->{v_W} = [];
   # N(0, 1/sqrt(d_model)) so that after InputEmbeddings' sqrt(d_model) multiplier,
   # output has std ~1, matching the scale of sinusoidal positional embeddings.
   my $prng = Math::Random::OO::Normal->new(0, 1.0 / sqrt($self->{embeddings}));
   foreach my $row ( 0 .. $self->{vocab_size} - 1) {
      foreach my $col ( 0 .. $self->{embeddings} - 1) {
         $self->{W}[$row][$col]   = $prng->next();
         $self->{m_W}[$row][$col] = 0;
         $self->{v_W}[$row][$col] = 0;
      }
   }
   $self->{adam_step} = 1;
}

sub set_weights {
   my $self = shift;
   my %args = @_;
   if (!defined($args{weights})) {
      $self->initialise_weights();
   } else {
      $self->{W} = $args{weights};
      # zero-init moments on reload (standard — Adam moments not serialised)
      $self->{m_W} = [];
      $self->{v_W} = [];
      foreach my $row ( 0 .. $self->{vocab_size} - 1) {
         foreach my $col ( 0 .. $self->{embeddings} - 1) {
            $self->{m_W}[$row][$col] = 0;
            $self->{v_W}[$row][$col] = 0;
         }
      }
      $self->{adam_step} = 1;
   }
}

sub get_weights {
   my $self = shift;
   return { vocab_size => $self->{vocab_size},
            embeddings => $self->{embeddings},
            weights => $self->{W} };
}

sub forward {
   my $self = shift;
   my %args = @_;
   my $batch = $args{batch};

   $self->{input} = dclone($batch);

   my $N = scalar(@{$batch}); # batch size
   $self->{batch_size} = $N;

   my $seq_length = scalar(@{$batch->[0]});
   $self->{seq_length} = $seq_length;


   my $output = [];

   foreach my $b (0 .. $N - 1) {
      my $out = [];
      foreach my $i (@{$self->{input}[$b]}) {
         push @$out, $self->{W}[$i];
      }
      $output->[$b] = dclone($out);
   }
   if ($self->{debug}) {
      foreach my $b(0 .. $N - 1) {
         print_2d_array("embeddng output $b", $output->[$b]);
      }
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

sub new {
   my $class = shift;
   my $self = {};
   my %args = @_;
   $self->{name} = $args{name} || "Embedding";
   my $debug = $args{debug};
   if (defined($debug) and $debug =~ /^\d+$/ and $debug > 0) { 
      $debug = 1;
   } else {
      $debug = 0;
   }
   $self->{debug} = $debug;

   my $vocab_size = $args{vocab_size};
   if (!defined($vocab_size) or $vocab_size !~ /^\d+$/) {
      die "Invalid insize parameter '$vocab_size', must be integer ";
   }
   $self->{vocab_size} = $vocab_size;
   my $embeddings = $args{embeddings};
   if (!defined($embeddings) or $embeddings !~ /^\d+$/) {
      die "Invalid embeddings parameter '$embeddings', must be integer ";
   }
   $self->{embeddings} = $embeddings;
   $self->{outsize} = $args{vocab_size} * $args{embeddings};
   say "ML::Embedding maximum outsize = " . $self->{outsize}  if $self->{debug};
   return bless $self, $class;
}  

1;
