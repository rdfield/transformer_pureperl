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
   my $next = $params{next};
   my $delta = $next->gradient();
   $self->{gradient} = [];
   if (ref($delta->[0][0]) eq 'ARRAY') {
      # 3D: [batch][seq][features]
      for my $b (0 .. $#$delta) {
         for my $i (0 .. $#{$delta->[$b]}) {
            for my $j (0 .. $#{$delta->[$b][$i]}) {
               $self->{gradient}[$b][$i][$j] = $self->{input}[$b][$i][$j] > 0 ? $delta->[$b][$i][$j] : 0;
            }
         }
      }
   } else {
      # 2D: [seq][features]
      for my $i (0 .. $#$delta) {
         for my $j (0 .. $#{$delta->[$i]}) {
            $self->{gradient}[$i][$j] = $self->{input}[$i][$j] > 0 ? $delta->[$i][$j] : 0;
         }
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
   $self->{input} = dclone($X);
   my $out = [];
   if (ref($X->[0][0]) eq 'ARRAY') {
      # 3D: [batch][seq][features]
      for my $b (0 .. $#$X) {
         for my $i (0 .. $#{$X->[$b]}) {
            for my $j (0 .. $#{$X->[$b][$i]}) {
               $out->[$b][$i][$j] = $X->[$b][$i][$j] > 0 ? $X->[$b][$i][$j] : 0;
            }
         }
      }
   } else {
      # 2D: [seq][features]
      for my $i (0 .. $#$X) {
         for my $j (0 .. $#{$X->[$i]}) {
            $out->[$i][$j] = $X->[$i][$j] > 0 ? $X->[$i][$j] : 0;
         }
      }
   }
   return $out;
}

1;
