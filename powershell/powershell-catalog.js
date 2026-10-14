/**
 * PowerShell Command Catalog & Slot-Filling Engine for S1stem
 * ============================================================
 * Provides rich idiomatic PowerShell patterns and dynamic parameter extraction
 * for system operations, file management, processes, networking, and Git.
 */

export const POWERSHELL_PATTERNS = [
  // --- File & Directory Creation ---
  {
    id: 'fs_create_file',
    label: 'Create New File',
    description: 'Create new file, command to create a new empty file in a folder or disk path, touch empty document New-Item -ItemType File.',
    template: (p) => {
      let target = p.path || p.targetDir || '';
      if (!target || target.endsWith('\\') || target.endsWith('/')) {
        target = (target ? target.replace(/[\\\/]$/, '') + '\\' : '') + (p.fileName || 'new_file.txt');
      } else if (!target.match(/\.[a-zA-Z0-9]+$/)) {
        target = target + '\\' + (p.fileName || 'new_file.txt');
      }
      return `New-Item -Path "${target}" -ItemType File -Force`;
    },
    explanation: (p) => `Creates a new file at the specified path using New-Item with File type.`,
    defaultRisk: 1
  },
  {
    id: 'fs_create_dir_and_enter',
    label: 'Create Directory and Navigate Inside',
    description: 'Create new directory folder and enter inside, make directory and navigate into cd, New-Item Directory Set-Location.',
    template: (p) => {
      const dir = p.dirName || p.path || 'new_dir';
      return `New-Item -ItemType Directory -Path "${dir}" -Force; Set-Location -Path "${dir}"`;
    },
    explanation: (p) => `Creates the specified folder and immediately navigates inside using Set-Location.`,
    defaultRisk: 1
  },
  {
    id: 'fs_create_directory',
    label: 'Create Directory / Folder',
    description: 'Create new directory folder, make directory folder mkdir New-Item -ItemType Directory.',
    template: (p) => `New-Item -ItemType Directory -Path "${p.dirName || p.path || 'new_folder'}" -Force`,
    explanation: (p) => `Creates a new directory at the specified path.`,
    defaultRisk: 1
  },

  // --- File Listing & Inspection ---
  {
    id: 'fs_list_files',
    label: 'List Files in Directory',
    description: 'List files in directory, command to get list of files in directory folder, show folder contents directory listing Get-ChildItem ls dir.',
    template: (p) => `Get-ChildItem -Path "${p.path || p.targetDir || '.'}"`,
    explanation: (p) => `Lists all files and subdirectories in the specified directory.`,
    defaultRisk: 1
  },
  {
    id: 'fs_list_files_detailed',
    label: 'List Files with Details or Hidden',
    description: 'List directory contents with details, show hidden files with sizes and dates, Get-ChildItem -Force Format-Table.',
    template: (p) => `Get-ChildItem -Path "${p.path || p.targetDir || '.'}" -Force | Format-Table Mode, Name, Length, LastWriteTime -AutoSize`,
    explanation: (p) => `Lists files including hidden ones with formatted sizes and modification dates.`,
    defaultRisk: 1
  },
  {
    id: 'fs_read_file_content',
    label: 'Read File Content',
    description: 'Read inspect file content, show print display file content to console, cat type Get-Content.',
    template: (p) => `Get-Content -Path "${p.path || p.fileName || 'file.txt'}"`,
    explanation: (p) => `Displays the contents of the specified file to the console.`,
    defaultRisk: 1
  },
  {
    id: 'fs_tail_log_file',
    label: 'Follow / Tail Live Log File',
    description: 'Follow live log file stream, monitor watch tail log file in real time tail -f, Get-Content -Wait -Tail.',
    template: (p) => `Get-Content -Path "${p.path || p.fileName || 'app.log'}" -Wait -Tail 50`,
    explanation: (p) => `Monitors the latest lines of a log file in real time.`,
    defaultRisk: 1
  },
  {
    id: 'fs_search_file_by_name',
    label: 'Search Files by Name or Extension',
    description: 'Search find files by name extension, recursively search files by name filter, Get-ChildItem -Filter -Recurse.',
    template: (p) => `Get-ChildItem -Path "${p.targetDir || '.'}" -Filter "*${p.filter || p.ext || p.fileName || ''}*" -Recurse -File -ErrorAction SilentlyContinue`,
    explanation: (p) => `Recursively searches for all files matching the specified filter.`,
    defaultRisk: 1
  },
  {
    id: 'fs_search_text_grep',
    label: 'Search Text Pattern in Files (Grep)',
    description: 'Search text pattern in files grep, find string keyword inside files text search, Select-String.',
    template: (p) => `Select-String -Path "${p.targetDir || '.'}\\*.*" -Pattern "${p.query || p.keyword || 'text'}" -CaseSensitive:$false`,
    explanation: (p) => `Searches for the specified pattern across files in the directory.`,
    defaultRisk: 1
  },

  // --- File Manipulation & Cleanup ---
  {
    id: 'fs_copy_item',
    label: 'Copy File or Folder',
    description: 'Copy file or folder destination, duplicate file or directory to another destination cp, Copy-Item.',
    template: (p) => `Copy-Item -Path "${p.source || p.path || 'file.txt'}" -Destination "${p.destination || 'copy_file.txt'}" -Recurse`,
    explanation: (p) => `Copies the specified file or directory to the target destination.`,
    defaultRisk: 2
  },
  {
    id: 'fs_move_item',
    label: 'Move or Rename File / Folder',
    description: 'Move rename file or folder, relocate file or folder destination mv, Move-Item.',
    template: (p) => `Move-Item -Path "${p.source || p.path || 'file.txt'}" -Destination "${p.destination || 'new_path'}"`,
    explanation: (p) => `Moves or renames the file or folder to the new destination.`,
    defaultRisk: 2
  },
  {
    id: 'fs_delete_file',
    label: 'Delete Specific File',
    description: 'Delete remove specific file, remove single file from disk rm del, Remove-Item.',
    template: (p) => `Remove-Item -Path "${p.path || p.fileName || 'file.txt'}"`,
    explanation: (p) => `Deletes the specified file from disk.`,
    defaultRisk: 4
  },
  {
    id: 'fs_delete_directory_recursive',
    label: 'Delete Directory Recursively (Destructive)',
    description: 'Delete directory recursively, remove folder recursively, delete all files and subfolders, Remove-Item -Recurse -Force rm -rf.',
    template: (p) => `Remove-Item -Path "${p.dirName || p.path || 'temp'}" -Recurse -Force`,
    explanation: (p) => `Recursively deletes the entire folder and all of its contents (irreversible action).`,
    defaultRisk: 5
  },

  // --- Navigation & Environment ---
  {
    id: 'nav_change_directory',
    label: 'Change Directory (cd)',
    description: 'Change directory navigate cd, navigate to folder path location, Set-Location cd.',
    template: (p) => `Set-Location -Path "${p.path || p.targetDir || '~'}"`,
    explanation: (p) => `Changes the current working directory.`,
    defaultRisk: 1
  },
  {
    id: 'nav_current_directory',
    label: 'Show Current Directory Path',
    description: 'Show print current directory path pwd, print working directory where am I, Get-Location.',
    template: () => `Get-Location`,
    explanation: (p) => `Prints the current directory path (pwd).`,
    defaultRisk: 1
  },
  {
    id: 'env_get_variable',
    label: 'Show Environment Variable',
    description: 'Show environment variable PATH USERNAME, inspect environment variable value, $env.',
    template: (p) => `$env:${p.varName || 'PATH'}`,
    explanation: (p) => `Displays the value of the specified environment variable.`,
    defaultRisk: 1
  },

  // --- Processes & Services ---
  {
    id: 'proc_list_processes',
    label: 'List Running Processes',
    description: 'List active running processes, show processes using the most memory RAM CPU top task manager, Get-Process ps.',
    template: (p) => p.procName
      ? `Get-Process -Name "${p.procName}" -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, CPU, WorkingSet64 -AutoSize`
      : `Get-Process | Sort-Object -Property WorkingSet64 -Descending | Select-Object -First 20 | Format-Table Id, ProcessName, @{Name='Mem(MB)';Expression={[math]::Round($_.WorkingSet64/1MB,1)}}, CPU -AutoSize`,
    explanation: (p) => `Lists running processes sorted by memory consumption.`,
    defaultRisk: 1
  },
  {
    id: 'proc_kill_process',
    label: 'Kill / Terminate Process',
    description: 'Kill terminate process, stop kill process by name or PID taskkill, Stop-Process.',
    template: (p) => p.pid
      ? `Stop-Process -Id ${p.pid} -Force`
      : `Stop-Process -Name "${p.procName || 'app'}" -Force`,
    explanation: (p) => `Forcefully terminates the specified process.`,
    defaultRisk: 4
  },
  {
    id: 'proc_start_process',
    label: 'Start Application or Process',
    description: 'Start application program, run launch executable application program, Start-Process.',
    template: (p) => `Start-Process "${p.appName || p.path || 'notepad.exe'}"`,
    explanation: (p) => `Starts the specified application or executable.`,
    defaultRisk: 2
  },

  // --- Networking & Web ---
  {
    id: 'net_test_connection_ping',
    label: 'Ping Host / Test Network Reachability',
    description: 'Ping network host reachability, ping google test network connection reachability ping, Test-Connection.',
    template: (p) => `Test-Connection -TargetName "${p.host || '8.8.8.8'}" -Count 4`,
    explanation: (p) => `Pings the host to verify network reachability and latency.`,
    defaultRisk: 1
  },
  {
    id: 'net_download_file',
    label: 'Download File from URL',
    description: 'Download file from URL web, download internet file curl wget, Invoke-WebRequest.',
    template: (p) => `Invoke-WebRequest -Uri "${p.url || 'https://example.com/file'}" -OutFile "${p.destination || 'downloaded_file'}"`,
    explanation: (p) => `Downloads the file from the specified URL and saves it to disk.`,
    defaultRisk: 2
  },
  {
    id: 'net_http_rest_api',
    label: 'Send HTTP REST API Request',
    description: 'Send HTTP REST API request, query REST API endpoint JSON response curl, Invoke-RestMethod.',
    template: (p) => `Invoke-RestMethod -Uri "${p.url || 'https://api.github.com'}" -Method Get`,
    explanation: (p) => `Sends an HTTP request to the API endpoint and automatically parses JSON response.`,
    defaultRisk: 1
  },
  {
    id: 'net_list_listening_ports',
    label: 'Show Active Listening Network Ports',
    description: 'Show active listening network ports, display listening TCP ports open connections netstat, Get-NetTCPConnection.',
    template: () => `Get-NetTCPConnection -State Listen | Select-Object LocalAddress, LocalPort, OwningProcess | Sort-Object LocalPort | Format-Table -AutoSize`,
    explanation: (p) => `Lists all active listening TCP ports on the local machine.`,
    defaultRisk: 1
  },

  // --- Git & Version Control ---
  {
    id: 'git_status_check',
    label: 'Check Git Status',
    description: 'Check git status repo, check repository status modified files git branch, git status.',
    template: () => `git status`,
    explanation: (p) => `Shows current Git working tree status and modified files.`,
    defaultRisk: 1
  },
  {
    id: 'git_commit_changes',
    label: 'Git Commit Modified Files',
    description: 'Git commit modified changes, record snapshot commit git save changes with message, git commit.',
    template: (p) => `git add -A; git commit -m "${p.commitMessage || 'Update changes'}"`,
    explanation: (p) => `Stages all modified files and creates a new Git commit.`,
    defaultRisk: 2
  },

  // --- System, Disk & Package Information ---
  {
    id: 'sys_disk_free_space',
    label: 'Check Disk Space',
    description: 'Check disk free space, inspect available drive disk space storage partitions, Get-PSDrive.',
    template: (p) => `Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}}, @{Name='Used(GB)';Expression={[math]::Round($_.Used/1GB,2)}} | Format-Table -AutoSize`,
    explanation: (p) => `Displays free and used disk space for all mounted drives.`,
    defaultRisk: 1
  },
  {
    id: 'sys_history_commands',
    label: 'View Command History',
    description: 'View command history, inspect recent console terminal executed commands, Get-History.',
    template: () => `Get-History -Count 25 | Format-Table Id, CommandLine, StartExecutionTime -AutoSize`,
    explanation: (p) => `Displays recent commands executed in the current PowerShell session.`,
    defaultRisk: 1
  },
  {
    id: 'sys_get_date_time',
    label: 'Get Current Date & Time',
    description: 'Get current date and time clock, what time is it what date is today timestamp calendar current time clock now, Get-Date.',
    template: () => `Get-Date`,
    explanation: (p) => `Shows the current system date and time using Get-Date.`,
    defaultRisk: 1
  },
  {
    id: 'sys_clear_screen',
    label: 'Clear Screen / Terminal',
    description: 'Clear screen terminal console, wipe clean terminal console cls clear, Clear-Host.',
    template: () => `Clear-Host`,
    explanation: (p) => `Clears the PowerShell terminal screen.`,
    defaultRisk: 1
  },
  {
    id: 'net_get_ip_address',
    label: 'Get IP Address / Network Config',
    description: 'Get local IP address network configuration, what is my local ip network adapter ipconfig, Get-NetIPAddress.',
    template: () => `Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch 'Loopback' } | Select-Object IPAddress, InterfaceAlias | Format-Table -AutoSize`,
    explanation: (p) => `Displays IPv4 addresses assigned to active network adapters.`,
    defaultRisk: 1
  }
];

/**
 * Intelligent Slot-Filling Parameter Extractor
 * Extracts real target paths (e.g. "D:\", "c:\test"), folder names ("new_dir"),
 * file names, URLs, PIDs, and keywords from natural language user input.
 * 
 * @param {string} prompt - Raw natural language user prompt
 * @returns {Object} Extracted slots / parameters
 */
export function extractSlots(prompt) {
  const slots = {};
  if (!prompt || typeof prompt !== 'string') return slots;

  const raw = prompt.trim();

  // 1. Windows Drive Letter or Full Path (e.g. "D:\", "C:\test", "D:\folder\file.txt", "d:/test")
  const driveMatch = raw.match(/\b([a-zA-Z]:[\\\/][a-zA-Z0-9_\-\.\ \\\/]*)/i);
  if (driveMatch) {
    slots.path = driveMatch[1].trim();
    // Normalize path separators
    slots.path = slots.path.replace(/\//g, '\\');
    if (slots.path.endsWith('\\') || slots.path.match(/^[a-zA-Z]:\\?$/i)) {
      slots.targetDir = slots.path;
    } else {
      const parts = slots.path.split('\\');
      slots.fileName = parts[parts.length - 1];
      slots.targetDir = parts.slice(0, -1).join('\\');
    }
  }

  // 2. Relative Directory or Path (e.g. "./test", "..\folder", "c:\test" if not caught above)
  if (!slots.path) {
    const relMatch = raw.match(/(\.[\/\\][a-zA-Z0-9_\-\.\/\\]+)/);
    if (relMatch) {
      slots.path = relMatch[1].trim();
      slots.targetDir = slots.path;
    }
  }

  // 3. Named directory / folder pattern:
  // e.g. "name of directory new_dir", "named new_dir", "called new_dir", "directory new_dir", "folder new_dir"
  const dirNamePatterns = [
    /(?:name\s+of\s+directory|name\s+of\s+folder|directory\s+name|folder\s+name|named|called|nome\s+della\s+directory|nome\s+della\s+cartella)\s+[:=]?\s*["']?([a-zA-Z0-9_\-]+)["']?/i,
    /(?:create|make|crea|nuova)\s+(?:a\s+)?(?:new\s+)?(?:directory|folder|cartella)\s+["']?([a-zA-Z0-9_\-]+)["']?/i,
    /(?:directory|folder|cartella)\s+["']?([a-zA-Z0-9_\-]+)["']?/i
  ];

  for (const regex of dirNamePatterns) {
    const match = raw.match(regex);
    if (match && match[1]) {
      const candidate = match[1].trim();
      // Avoid common false positives
      if (!['in', 'and', 'inside', 'new', 'a', 'an', 'the', 'file', 'directory', 'folder', 'ed', 'dentro', 'nuova', 'un', 'una', 'cartella'].includes(candidate.toLowerCase())) {
        slots.dirName = candidate;
        break;
      }
    }
  }

  // 4. File name pattern (e.g. "file test.txt", "file.json", "called app.log")
  const fileMatch = raw.match(/([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,5})/i);
  if (fileMatch) {
    slots.fileName = fileMatch[1];
    if (slots.targetDir) {
      slots.path = slots.targetDir.replace(/\\$/, '') + '\\' + slots.fileName;
    }
  }

  // 5. URL pattern (e.g. "https://...", "http://...")
  const urlMatch = raw.match(/(https?:\/\/[^\s"']+)/i);
  if (urlMatch) {
    slots.url = urlMatch[1];
  }

  // 6. PID pattern (e.g. "PID 8421", "id 1234")
  const pidMatch = raw.match(/\b(?:pid|id)\s+([0-9]+)\b/i);
  if (pidMatch) {
    slots.pid = parseInt(pidMatch[1], 10);
  }

  // 7. Process name pattern (e.g. "process chrome", "kill notepad")
  const procMatch = raw.match(/(?:process|processo)\s+["']?([a-zA-Z0-9_\-]+)["']?/i);
  if (procMatch) {
    slots.procName = procMatch[1];
  }

  // 8. Commit message pattern
  const commitMsgMatch = raw.match(/(?:message|messaggio)\s+[:=]?\s*["']([^"']+)["']/i);
  if (commitMsgMatch) {
    slots.commitMessage = commitMsgMatch[1];
  }

  return slots;
}
