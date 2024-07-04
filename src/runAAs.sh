#!/bin/bash

# Specify data set
DATA_NAME=$1

# Directory paths
SUBSET_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/data/subsetsDataPKLs"
PBS_SCRIPT_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/jobScripts"
SCRIPT_PATH="/scratch/q27/jt5911/iterative-archetypal-analysis/PIAA.py"

# Create directory for PBS scripts if it doesn't exist
mkdir -p $PBS_SCRIPT_DIR

# Create a list of all subset files
cd $SUBSET_DIR
SUBSET_FILES=($(ls $DATA_NAME*.pkl))

# Generate and submit a PBS script for each subset file
cd ../../
echo "Running archetypal analysis for all subsets..."
for subset_file in "${SUBSET_FILES[@]}"; do
    pbs_script="${PBS_SCRIPT_DIR}/${subset_file%.pkl}.pbs"
    
    # Create PBS script
    cat > "$pbs_script" << EOL
#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=$3,walltime=$4,mem=$5GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.10.4

cd \$PBS_O_WORKDIR
python3 $SCRIPT_PATH $SUBSET_DIR/$subset_file $2

EOL

    # Submit the PBS script
    qsub "$pbs_script"
    
    echo "  Submitted job for $subset_file"
done

# echo -e "\nAll jobs submitted!"
