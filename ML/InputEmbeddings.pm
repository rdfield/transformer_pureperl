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
