# Create 100 LSVM worlds - No Global bidders.
a=0
while [ $a -lt 100 ]; do
  # $1 should contain the path to noisy_combinatorial_markets.jar.
  java -jar "$1" worlds/world$a.json
  a=$(expr $a + 1)
done
