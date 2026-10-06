#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#if defined(__has_include)
#  if __has_include(<linux/landlock.h>)
#    include <linux/landlock.h>
#    define HAVE_LANDLOCK_H 1
#  endif
#endif
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <linux/audit.h>
#include <stddef.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef HAVE_LANDLOCK_H
/* Kernel headers predate Landlock (for example RHEL 8, kernel 4.18). These minimal
   UAPI definitions keep the helper buildable; the run-time ABI probe decides whether
   Landlock is actually used. */
#include <linux/types.h>
struct landlock_ruleset_attr { __u64 handled_access_fs; };
#define LANDLOCK_CREATE_RULESET_VERSION (1U << 0)
enum landlock_rule_type { LANDLOCK_RULE_PATH_BENEATH = 1 };
struct landlock_path_beneath_attr { __u64 allowed_access; __s32 parent_fd; } __attribute__((packed));
#define LANDLOCK_ACCESS_FS_EXECUTE    (1ULL << 0)
#define LANDLOCK_ACCESS_FS_WRITE_FILE (1ULL << 1)
#define LANDLOCK_ACCESS_FS_READ_FILE  (1ULL << 2)
#define LANDLOCK_ACCESS_FS_READ_DIR   (1ULL << 3)
#endif
#ifndef __NR_clone3
#define __NR_clone3 435
#endif
#ifndef __NR_close_range
#define __NR_close_range 436
#endif
#ifndef __NR_landlock_create_ruleset
#define __NR_landlock_create_ruleset 444
#endif
#ifndef __NR_landlock_add_rule
#define __NR_landlock_add_rule 445
#endif
#ifndef __NR_landlock_restrict_self
#define __NR_landlock_restrict_self 446
#endif

static void die(const char *s) { perror(s); exit(126); }
static uint64_t ro = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_EXECUTE;
static void allow(int fd, const char *path, uint64_t rights, int required) {
    if(fd<0) return; /* Landlock unavailable on this kernel: no ruleset to add to */
    int p=open(path,O_PATH|O_CLOEXEC);
    if(p<0) { if(required) die(path); return; }
    struct stat st; if(fstat(p,&st)) die("fstat");
    if(!S_ISDIR(st.st_mode)) rights &= LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_EXECUTE;
    struct landlock_path_beneath_attr rule={.allowed_access=rights,.parent_fd=p};
    if(syscall(__NR_landlock_add_rule,fd,LANDLOCK_RULE_PATH_BENEATH,&rule,0)) die(path);
    close(p);
}
static void offline(int worker) {
    struct sock_filter filter[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_socket,0,1),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_ptrace,0,1),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_process_vm_readv,0,1),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_process_vm_writev,0,1),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW)
    };
    struct sock_fprog p={.len=sizeof(filter)/sizeof(filter[0]),.filter=filter};
    if(prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&p)) die("seccomp");
    if(worker) {
        /* Candidate code cannot create background processes or signal other jobs.
           Native library threads remain possible via CLONE_THREAD. */
        struct sock_filter child_filter[]={
            BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_fork,0,1),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_vfork,0,1),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_clone3,0,1),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|ENOSYS),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_kill,0,1),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_clone,0,4),
            BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
            BPF_JUMP(BPF_JMP|BPF_JSET|BPF_K,CLONE_THREAD,1,0),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW)
        };
        struct sock_fprog q={.len=sizeof(child_filter)/sizeof(child_filter[0]),.filter=child_filter};
        if(prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&q))die("worker seccomp");
    }
}
int main(int argc,char **argv) {
    if(argc<4) { fprintf(stderr,"usage: confine ROOT author COMMAND ... | confine ROOT worker SNAPSHOT COMMAND ...\n");return 2; }
    char *root=realpath(argv[1],NULL); if(!root||!strcmp(root,"/")) die("root");
    int worker=!strcmp(argv[2],"worker");
    if((!worker&&strcmp(argv[2],"author"))||(worker&&argc<6)) return 2;
    int abi=syscall(__NR_landlock_create_ruleset,NULL,0,LANDLOCK_CREATE_RULESET_VERSION);
    /* Landlock ABI >=3 (Linux >=6.2) gives full filesystem confinement. On older kernels
       the filesystem rules are skipped with a warning; seccomp, no_new_privs, fd/env
       scrubbing and the working-directory contract below still apply. */
    int landlock=abi>=3;
    uint64_t all=(1ULL<<15)-1;
    struct landlock_ruleset_attr rules={.handled_access_fs=all};
    int fd=-1;
    if(landlock) { fd=syscall(__NR_landlock_create_ruleset,&rules,sizeof(rules),0);if(fd<0)die("ruleset"); }
    else fprintf(stderr,"confine: warning: Landlock ABI %d < 3 on this kernel; filesystem confinement disabled, "
        "seccomp and environment isolation still applied\n",abi<0?0:abi);
    char p[8192];
    if(worker) {
        allow(fd,argv[3],ro,1);
        const char *parts[]={"/.venv","/runtime/python","/runtime/tools",NULL};
        for(const char **s=parts;*s;s++){snprintf(p,sizeof(p),"%s%s",root,*s);allow(fd,p,ro,1);}
    } else {
        allow(fd,root,ro,1);
        const char *parts[]={"/results","/.cache","/.requests",NULL};
        for(const char **s=parts;*s;s++){snprintf(p,sizeof(p),"%s%s",root,*s);allow(fd,p,all,1);}
    }
    const char *system[]={"/usr/bin","/usr/lib","/usr/lib64","/usr/share","/bin","/lib","/lib64",
        "/etc/ld.so.cache","/etc/ld.so.conf","/etc/localtime","/proc/self","/proc/thread-self","/proc/cpuinfo","/proc/meminfo",
        "/proc/sys/kernel/osrelease","/sys/devices/system/cpu","/sys/devices/system/node",NULL};
    for(const char **s=system;*s;s++)allow(fd,*s,ro,0);
    const char *devices[]={"/dev/null","/dev/zero","/dev/random","/dev/urandom",NULL};
    for(const char **s=devices;*s;s++)allow(fd,*s,LANDLOCK_ACCESS_FS_READ_FILE|LANDLOCK_ACCESS_FS_WRITE_FILE,1);
    if(prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0))die("no_new_privs");
    if(landlock) { if(syscall(__NR_landlock_restrict_self,fd,0))die("restrict"); close(fd); }
    if(syscall(__NR_close_range,3,~0U,0)&&errno!=ENOSYS)die("close_range");
    if(clearenv())die("clearenv");
    snprintf(p,sizeof(p),"%s/.venv/bin:/usr/bin:/bin",root);setenv("PATH",p,1);
    snprintf(p,sizeof(p),"%s/.cache/tmp",root);setenv("TMPDIR",p,1);
    snprintf(p,sizeof(p),"%s/runtime/tools",root);setenv("PYTHONPATH",p,1);
    setenv("PYTHONNOUSERSITE","1",1);setenv("PYTHONDONTWRITEBYTECODE","1",1);
    setenv("PYTHONUNBUFFERED","1",1);setenv("OMP_NUM_THREADS","1",1);
    setenv("OPENBLAS_NUM_THREADS","1",1);setenv("MKL_NUM_THREADS","1",1);
    setenv("LANG","C.UTF-8",1);setenv("GIT_CONFIG_NOSYSTEM","1",1);setenv("GIT_CONFIG_GLOBAL","/dev/null",1);
    if(chdir(worker?argv[3]:root))die("chdir");
    offline(worker);
    execvp(argv[worker?4:3],argv+(worker?4:3));die("execvp");
}
