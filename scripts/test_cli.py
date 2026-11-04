"""mini-lla 的 CLI 冒烟测试:检查退出码和关键输出,不比对随机模型的文本质量。"""
import os
import subprocess
import sys

BINARY = os.environ.get("MINI_LLA_BIN", os.path.join("build", "Debug", "mini-lla.exe"))


def run(arguments, stdin=""):
    """运行一次 CLI,返回 (返回码, 合并后的输出)。"""
    result = subprocess.run(
        [BINARY] + arguments, input=stdin, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def check(name, condition):
    """打印一条断言结果,返回失败计数。"""
    print(("[ ok ] " if condition else "[fail] ") + name)
    return 0 if condition else 1


def main():
    """依次检查生成、帮助、参数校验和聊天退出。"""
    failures = 0

    code, output = run(["generate", "--model", "models/tiny", "-n", "4"])
    failures += check("generate 正常返回", code == 0)
    failures += check("generate 输出 token 统计", "prompt tokens:" in output)

    code, output = run(["help"])
    failures += check("help 正常返回", code == 0)
    failures += check("help 打印用法", "usage:" in output)

    code, output = run(["generate", "--model", "models/tiny", "-n", "-1"])
    failures += check("负的 -n 被拒绝", code != 0)
    failures += check("错误信息包含参数名", "--n-predict" in output)

    code, output = run(["generate", "--model", "models/tiny", "--bogus"])
    failures += check("未知参数被拒绝", code != 0)

    code, output = run(["generate", "--model", "no-such-model", "-n", "1"])
    failures += check("缺失模型返回非零", code != 0)

    code, output = run(["run", "--model", "models/tiny", "-n", "2"], "/exit\n")
    failures += check("run 可以正常退出", code == 0)

    code, output = run(["inspect", "--model", "models/tiny"])
    failures += check("inspect 打印配置", code == 0 and "n_layers" in output)

    code, output = run(["bench", "--model", "models/tiny", "-n", "2",
                        "--threads", "2"])
    failures += check("bench 正常返回", code == 0)
    failures += check("bench 打印吞吐", "tokens/s" in output)
    failures += check("bench 使用 2 个线程", "threads: 2" in output)

    code, output = run(["--cuda-info"])
    failures += check("cuda-info 正常返回", code == 0)
    failures += check("cuda-info 说明 CUDA 状态", "CUDA" in output)

    code, output = run(["generate", "--model", "models/tiny", "--backend", "cuda"])
    failures += check("未启用 CUDA 时请求 cuda 后端被拒绝", code != 0)

    print("failures=%d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
