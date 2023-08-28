import subprocess
numBlocks_values = [1,2,3,4,5,6,7,8]
blockSize_values = [64, 128, 256, 512, 1024]


program_file = "copy_bw.cu"
with open(program_file, "r") as f:
    source_code = f.read()

for numBlocks in numBlocks_values:
    for blockSize in blockSize_values:
        # 使用字符串替换修改源代码中的 numBlocks 和 blockSize 值
        modified_code = source_code.replace("int numBlocks =", f"int numBlocks = {numBlocks}")
        modified_code = modified_code.replace("int blockSize =", f"int blockSize = {blockSize}")

        # 将修改后的源代码写入临时文件
        with open("temp_copy_bw.cu", "w") as f:
            f.write(modified_code)

        # 编译和执行程序
        compile_command = "sudo /usr/local/cuda/bin/nvcc temp_copy_bw.cu -o test1"
        execution_command = "./test1"

        subprocess.run(compile_command, shell=True)
        result = subprocess.run(execution_command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()  # 获取输出结果

        print(output)
        print("------------------")

