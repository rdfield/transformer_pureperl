package ML::InputEmbeddings;
use lib '/home/mvine/embeddings_mk2';
use ML::Embedding;
use Modern::Perl;
use lib '/home/mvine/imagenet';
use ML::Util qw(print_2d_array mult_constant);
use Storable qw(dclone);
use Carp qw(cluck);

sub get_grad_tensors {
   my $self = shift;
   return $self->{embedding}->get_grad_tensors();
}

sub optimise {
   my $self = shift;
   $self->{embedding}->optimise( @_ );
}

sub get_weights {
   my $self = shift;
   return $self->{embedding}->get_weights();
}

sub set_weights {
   my $self = shift;
   $self->{embedding}->set_weights(@_);
}

sub new {
   my $class = shift;
   my %args = @_;
   my $self = {};
   $self->{embeddings} = $args{embeddings};
   $self->{vocab_size} = $args{vocab_size};
   $self->{embedding} = ML::Embedding->new(vocab_size => $self->{vocab_size}, embeddings => $self->{embeddings});
   $self->{embedding}->set_weights( weights => $args{weights} );
   return bless $self, $class;
}

sub forward {
   my $self = shift;
   my %args = @_;
   my $input = dclone($args{batch});
   my $output = $self->{embedding}->forward(batch => $input);
   my $batch_size = scalar(@$output);
   if ($self->{debug}) {
      say "output from embedding batch_size $batch_size";
   }
   foreach my $b (0 .. $batch_size - 1) {
      if ($self->{debug}) {
         print_2d_array("output $b", $output->[$b]);
         say "multiplier = " . sqrt($self->{embeddings});
      }
      mult_constant($output->[$b], sqrt($self->{embeddings}));
      if ($self->{debug}) {
         print_2d_array("input embed output $b", $output->[$b]);
      }
   }
   if ($ENV{GRAD_PROBE}) {
      my $sumsq = 0;
      for my $b (@$output) { for my $r (@$b) { $sumsq += $_ * $_ for @$r } }
      my $wsq = 0;
      for my $r (@{$self->{embedding}{W}}) { $wsq += $_ * $_ for @$r }
      printf "    [embed_fwd] |out|=%.3e  |W|=%.3e  vocab=%d emb=%d  bs=%d seq=%d\n",
             sqrt($sumsq), sqrt($wsq), scalar(@{$self->{embedding}{W}}),
             scalar(@{$self->{embedding}{W}[0]}),
             scalar(@$output), scalar(@{$output->[0]});
   }
   return $output;
}

sub gradient {
   my $self = shift;
   return $self->{embedding}->gradient();
}

sub backward {
   my $self = shift;
   my %params = @_;
   my $next = $params{ next };
   my $in_grad = $next->gradient();
   # Forward multiplied output by sqrt(embeddings), so chain rule requires
   # the same scale factor on the incoming gradient before passing to Embedding.
   my $scale = sqrt($self->{embeddings});
   my $scaled = dclone($in_grad);
   for my $b (@$scaled) { for my $row (@$b) { $_ *= $scale for @$row } }
   $self->{weights_grad} = $scaled;
   {
      package ML::InputEmbeddings::ScaledGrad;
      sub new      { bless { g => $_[1] }, $_[0] }
      sub gradient { $_[0]->{g} }
   }
   $self->{embedding}->backward(
       next => ML::InputEmbeddings::ScaledGrad->new($scaled));
}
1;
