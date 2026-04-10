package ML::LossGradient;
use Modern::Perl;

sub new {
   my $class = shift;
   my %self = @_;
   return bless \%self, $class;
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

1;
