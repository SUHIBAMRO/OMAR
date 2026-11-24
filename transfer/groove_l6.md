# Job `groove_l6`: step by step

1. **On your computer**, download (opens in the browser, lands in Downloads):
   https://github.com/SUHIBAMRO/OMAR/raw/claude/claude-code-question-d307wp/transfer/groove_l6.tar.gz
2. **On your computer**, in a new PowerShell window:
   ```
   scp -O $HOME\Downloads\groove_l6.tar.gz <login>@cslogin.tu-ilmenau.de:~/
   ```
3. **On the cluster** (after `ssh <login>@cslogin.tu-ilmenau.de`), one line at a time:
   ```
   md5sum ~/groove_l6.tar.gz
   ```
   It must print `7c2711606c28383988c0861189ee4b49`. Then:
   ```
   cd ~/OMAR && tar xzf ~/groove_l6.tar.gz && cd Practical_Examples
   bsub -gpu "num=1:mode=exclusive_process" -q BatchGPU -R "span[hosts=1] rusage[mem=64000]" -n 8 -W 18:00 -o groove_l6.%J.out -e groove_l6.%J.err < groove_l6_job.sh
   ```
   Expected: `Job <number> is submitted to queue <BatchGPU>.`
4. **When `bjobs` no longer lists it**, for each output:
   ```
   cd ~/OMAR/Practical_Examples/omar_pfem/verification_agent
   md5sum <output>
   cat <output>
   ```
   and paste both into the chat. Outputs: iga_b3_l6_result.json
