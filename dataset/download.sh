mkdir -p dataset/CMU_1/RAW
cd dataset/CMU_1/RAW
wget -c https://www.dropbox.com/s/02j1bytymufmpgw/Trj1.zip
unzip Trj1.zip
mv Traj1/Trj1/1/* .
rm -r Traj1*
   
mkdir -p dataset/PITT/DATA
cd dataset/PITT/DATA
wget -c https://www.dropbox.com/sh/9bwir43a8mqk6nq/AAA9jliF61inZlKlqS6jw_9-a?dl=0
unzip AAA9jliF61inZlKlqS6jw_9-a?dl=0
rm -r AAA9jliF61inZlKlqS6jw_9-a?dl=0
cd ../../..

mkdir -p results/
cd results/
wget -c https://www.dropbox.com/sh/89w9ptm0uqurimn/AACPfu_JFkMWdRaRpVD-Kvuea?dl=0
unzip AACPfu_JFkMWdRaRpVD-Kvuea?dl=0
rm -r AACPfu_JFkMWdRaRpVD-Kvuea?dl=0
cd ../../
