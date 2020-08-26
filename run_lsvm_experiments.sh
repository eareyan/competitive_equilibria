sudo apt-get install -y git
sudo apt-get install -y zip
sudo apt-get install -y screen
git clone https://eareyan@github.com/eareyan/competitive_equilibria.git
cd competitive_equilibria
sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
curl -O https://download.java.net/java/GA/jdk14/076bab302c7b4508975440c56f6cc26a/36/GPL/openjdk-14_linux-x64_bin.tar.gz
tar xvf openjdk-14_linux-x64_bin.tar.gz
sudo mv jdk-14 /opt/
sudo tee /etc/profile.d/jdk14.sh <<EOF
export JAVA_HOME=/opt/jdk-14
export PATH=\$PATH:\$JAVA_HOME/bin
EOF
source /etc/profile.d/jdk14.sh
mkdir output
python value_models.py output/
