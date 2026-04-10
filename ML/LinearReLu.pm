package ML::LinearReLu;
use Modern::Perl;
use Storable qw(dclone);
use ML::Util qw(transpose);

sub type {
   return "ML::LinearReLu";
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub optimise {
# nothing to optimise
}

sub backward {
   my $self = shift;
   my %params = @_;
   my $next = $params{ next };
   my $delta = $next->gradient();
   $self->{gradient} = [];
   #foreach my $i (0 .. $self->{insize} - 1) {
   #   foreach my $j (0 .. $self->{outsize} - 1) {
   foreach my $i (0 .. scalar(@$delta) - 1) {
      foreach my $j (0 .. scalar(@{$delta->[0]}) - 1) {
         $self->{gradient}[$i][$j] = ($self->{input}[$i][$j] > 0? $delta->[$i][$j] : 0); 
      }
   }
}


sub new {
   my $class = shift;
   my $self = {};
   my %config = @_;
   $self->{insize} = $config{insize};
   $self->{outsize} = $config{insize};
   $self->{name} = $config{name} || "linearrelu";
   return bless $self, $class;
}

sub outsize {
  my $self = shift;
  return $self->{outsize};
}

sub name {
  my $self = shift;
  return $self->{name};
}

sub initialise {
   my $self = shift;
# nothing to initialise at this point, no parameters to learn for Relu
}

sub forward {
   my $self = shift;
   my %args = @_;
   my $X = $args{batch};
   my $insize = scalar(@{$X}); 
   my $outsize = scalar(@{${X}->[0]}); 
   $self->{input} = dclone($X);
   my $out = [];
   foreach my $i (0 .. $insize - 1) {
      foreach my $j (0 .. $outsize - 1) {
         $out->[ $i ][ $j ] = $X->[ $i ][ $j ] > 0?$X->[ $i ][ $j ]:0;
      }
   } 

   return $out;
}

1;
