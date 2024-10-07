process CSVCombiner{
    time '5h'
    label "pytorch"
    input: 
    file g
    val out_file_prefix
    
    script: 
    """
    python3 "$projectDir/bin/csvCombiner.py" "${params.results_dir_path}${out_file_prefix}.csv"
    """
} 


