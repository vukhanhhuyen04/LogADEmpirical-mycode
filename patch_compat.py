python3 << 'EOF'
import subprocess

# Lỗi np.float bị NumPy mới xóa (ảnh hưởng PLELog)
for f in ["logadempirical/PLELog/data/Instance.py",
          "logadempirical/PLELog/data/Embedding.py",
          "logadempirical/PLELog/pipeline.py"]:
    subprocess.run(["sed", "-i", "", "s/dtype=np\\.float)/dtype=float)/g", f])  # macOS sed cần "" sau -i

# Lỗi chia cho 0 khi model không dự đoán bất thường nào
path = "logadempirical/logdeep/tools/predict.py"
with open(path) as f:
    content = f.read()
old = '''        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)
        with open(self.output_dir + self.model_name + "-leadtime.txt", mode="w") as f:
            [f.write(str(i) + "\\n") for i in lead_time]
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}, '
              'Lead time: {:.3f}'.format(P, R, F1, SP, sum(lead_time) / len(lead_time)))'''
new = '''        P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
        SP = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        with open(self.output_dir + self.model_name + "-leadtime.txt", mode="w") as f:
            [f.write(str(i) + "\\n") for i in lead_time]
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        avg_lead_time = sum(lead_time) / len(lead_time) if len(lead_time) > 0 else 0.0
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}, '
              'Lead time: {:.3f}'.format(P, R, F1, SP, avg_lead_time))'''
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)

# Lỗi model.cuda() hardcode trong PLELog (không tự nhận biết máy không có CUDA)
path2 = "logadempirical/PLELog/approaches/RNN_pipeline.py"
with open(path2) as f:
    c2 = f.read()
c2 = c2.replace(
    "    # if config.use_cuda:\n    model = model.cuda()",
    "    # if config.use_cuda:\n    if torch.cuda.is_available():\n        model = model.cuda()"
)
with open(path2, "w") as f:
    f.write(c2)

print("Đã vá xong 3 lỗi.")
EOF