# Job `ladder_nh_plate_hole_3d_pe_P232`: step by step

1. **On your computer**, download (opens in the browser, lands in Downloads):
   https://github.com/SUHIBAMRO/OMAR/raw/claude/claude-code-question-d307wp/transfer/ladder_nh_plate_hole_3d_pe_P232.tar.gz
2. **On your computer**, in a new PowerShell window:
   ```
   scp -O $HOME\Downloads\ladder_nh_plate_hole_3d_pe_P232.tar.gz <login>@cslogin.tu-ilmenau.de:~/
   ```
3. **On the cluster** (after `ssh <login>@cslogin.tu-ilmenau.de`), one line at a time:
   ```
   md5sum ~/ladder_nh_plate_hole_3d_pe_P232.tar.gz
   ```
   It must print `7ed0fa16b5cb84af7cf54dddc92fbcb7`. Then:
   ```
   cd ~/OMAR && tar xzf ~/ladder_nh_plate_hole_3d_pe_P232.tar.gz && cd Practical_Examples
   bsub -gpu "num=1:mode=exclusive_process" -q BatchGPU -R "span[hosts=1] rusage[mem=120000]" -n 8 -W 24:00 -o ladder_nh_plate_hole_3d_pe_P232.%J.out -e ladder_nh_plate_hole_3d_pe_P232.%J.err < ladder_nh_plate_hole_3d_pe_P232_job.sh
   ```
   Expected: `Job <number> is submitted to queue <BatchGPU>.`
4. **When `bjobs` no longer lists it**, run this one line and paste everything it prints into
   the chat:
   ```
   cd ~/OMAR/Practical_Examples/omar_pfem/verification_agent && for f in cluster_jobs/ladder_nh_plate_hole_3d_pe_P232_result.json; do echo "=== $(basename $f) $(md5sum < $f | cut -c1-32)"; cat $f; echo; echo "=== end"; done
   ```
   (Outputs: cluster_jobs/ladder_nh_plate_hole_3d_pe_P232_result.json. The chat side rebuilds each file, checks its md5 and imports it.)
