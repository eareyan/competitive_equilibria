# Install command line tools needed.
sudo apt-get install -y git
sudo apt-get install -y zip
sudo apt-get install -y screen
sudo apt-get install -y python3-venv

# Clone the repo
git clone https://eareyan@github.com/eareyan/competitive_equilibria.git
cd competitive_equilibria

# Create virtual environment and install requirements.
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install java 14.
curl -O https://download.java.net/java/GA/jdk14/076bab302c7b4508975440c56f6cc26a/36/GPL/openjdk-14_linux-x64_bin.tar.gz
tar xvf openjdk-14_linux-x64_bin.tar.gz
sudo mv jdk-14 /opt/
sudo tee /etc/profile.d/jdk14.sh <<EOF
export JAVA_HOME=/opt/jdk-14
export PATH=\$PATH:\$JAVA_HOME/bin
EOF
source /etc/profile.d/jdk14.sh

# Create output directory and run experiments.
screen -dm bash -c 'python value_models.py GSVM output/GSVM/'
screen -dm bash -c 'python value_models.py LSVM output/LSVM/'

