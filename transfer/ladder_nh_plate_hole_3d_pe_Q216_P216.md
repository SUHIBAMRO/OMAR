# Job `ladder_nh_plate_hole_3d_pe_Q216_P216`: step by step

1. **On your computer**, download (opens in the browser, lands in Downloads):
   https://github.com/SUHIBAMRO/OMAR/raw/claude/claude-code-question-d307wp/transfer/ladder_nh_plate_hole_3d_pe_Q216_P216.tar.gz
2. **On your computer**, in a new PowerShell window:
   ```
   scp -O $HOME\Downloads\ladder_nh_plate_hole_3d_pe_Q216_P216.tar.gz <login>@cslogin.tu-ilmenau.de:~/
   ```
3. **On the cluster** (after `ssh <login>@cslogin.tu-ilmenau.de`), one line at a time:
   ```
   md5sum ~/ladder_nh_plate_hole_3d_pe_Q216_P216.tar.gz
   ```
   It must print `fe2bb7d61563bc4ed9507b1631653290`. Then:
   ```
   cd ~/OMAR && tar xzf ~/ladder_nh_plate_hole_3d_pe_Q216_P216.tar.gz && cd Practical_Examples
   bsub -gpu "num=1:mode=exclusive_process" -q BatchGPU -R "span[hosts=1] rusage[mem=120000]" -n 8 -W 24:00 -o ladder_nh_plate_hole_3d_pe_Q216_P216.%J.out -e ladder_nh_plate_hole_3d_pe_Q216_P216.%J.err < ladder_nh_plate_hole_3d_pe_Q216_P216_job.sh
   ```
   Expected: `Job <number> is submitted to queue <BatchGPU>.`
4. **When `bjobs` no longer lists it**, for each output:
   ```
   cd ~/OMAR/Practical_Examples/omar_pfem/verification_agent
   md5sum <output>
   cat <output>
   ```
   and paste both into the chat. Outputs: cluster_jobs/ladder_nh_plate_hole_3d_pe_Q216_P216_result.json
