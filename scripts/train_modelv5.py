import os
import sys
import shutil

# 添加路徑以導入模塊
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from core.data_manager import DataManager
from core.model_trainerv5 import ModelTrainer

def main():
    # 交互式選擇模式
    mode_input = input("請問要進入哪一個模型 1.重頭訓練 2.新資料訓練 3.弱類別針對訓練: ").strip()
    
    if mode_input == '1':
        mode = 'from_scratch'
    elif mode_input == '2':
        mode = 'incremental'
    elif mode_input == '3':
        mode = 'fine_tune_weak'
    else:
        print("❌ 無效輸入，請選擇 1, 2 或 3。")
        return

    print(f"🎯 開始模型訓練流程 (模式: {mode})")

    # Early Stopping 預設
    early_stopping_patience = 3
    early_stopping_monitor = 'test_acc'
    print(f"🛑 Early Stopping 設置: patience={early_stopping_patience}, monitor={early_stopping_monitor}")

    # --- 2. 基礎配置 ---
    config = BaseConfig()
    data_manager = DataManager(config)
    data_manager.create_fixed_train_test_split(train_ratio=0.8, force_resplit=False)
    print("✅ 固定訓練/測試分割已創建或確認")

    feature_samples = data_manager.get_all_feature_samples()
    if not feature_samples:
        print("❌ 沒有找到特徵文件。請確保已使用 'extract_features_finetuned.py' 生成新的 .npy 檔案。")
        return

    print(f"📊 使用 {len(feature_samples)} 個【高質量】特徵進行訓練")
    all_animators = list(data_manager.label_to_idx.keys())
    print(f"🎨 所有原畫師: {all_animators}")

    # --- 3. 根據不同模式執行不同邏輯 ---

    if mode == 'from_scratch':
        print("\n==============================")
        print("🔥 模式: 從頭訓練 (From Scratch)")
        print("   - 目標: 驗證新特徵的潛力上限")
        print("==============================")

        trainer = ModelTrainer(
            config,
            data_manager,
            use_incremental=False,
            use_kd=False,
            new_animators=None,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor
        )
        
        model, history = trainer.train_model()
        
        print("\n🏁 從頭訓練完成!")
        print("   - 這個模型的最佳準確率，代表了您當前數據和架構的性能天花板。")
        print("   - best_model.pth 和 latest_model.pth 已更新為此模型的權重。")

    elif mode == 'incremental':
        print("\n==============================")
        print("🧠 模式: 增量學習 (Incremental)")
        print("   - 目標: 在保留舊知識的基礎上，高效學習新類別")
        print("==============================")
        
        old_model_path = os.path.join(config.model_dir, 'best_model.pth')
        teacher_model_path = os.path.join(config.model_dir, 'train_model.pth')

        if not os.path.exists(old_model_path):
            print(f"❌ 錯誤: 增量學習需要一個代表舊知識的模型。")
            print(f"   請提供一個在舊類別上訓練好的模型，並將其命名為 '{os.path.basename(old_model_path)}' 放在 'models' 資料夾中。")
            return
            
        print(f"📚 找到舊模型 {old_model_path}，將其作為本次增量學習的 Teacher。")
        shutil.copyfile(old_model_path, teacher_model_path)

        # 預設新類別
        new_animators = ['吉成曜','田中宏紀']
        
        trainer = ModelTrainer(
            config,
            data_manager,
            use_incremental=True,
            replay_ratio=0.7,
            lambda_kd=1.0,
            kd_temperature=2.5,
            head_warmup_epochs=5,
            new_animators=new_animators,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor
        )
        
        model, history = trainer.train_model()

        print("\n🏁 增量學習訓練完成!")

    elif mode == 'fine_tune_weak':
        print("\n==============================")
        print("🛠️ 模式: 針對弱類別微調 (Fine-Tune Weak Classes)")
        print("   - 目標: 優化特定原畫師的性能，保留整體知識")
        print("==============================")

        # 預設弱類別
        weak_animators = ['國弘昌之','山下宏幸']
        print(f"🖌️ 預設弱類別: {weak_animators}")

        if not weak_animators:
            print("❌ 錯誤: fine_tune_weak 模式需要指定弱類別")
            return

        invalid_animators = [anim for anim in weak_animators if anim not in all_animators]
        if invalid_animators:
            print(f"❌ 錯誤: 以下原畫師不在數據集中: {invalid_animators}")
            return

        old_model_path = os.path.join(config.model_dir, 'best_model.pth')
        teacher_model_path = os.path.join(config.model_dir, 'teacher_model.pth')
        if not os.path.exists(old_model_path):
            print(f"❌ 錯誤: 需要現有模型 {old_model_path}")
            return
        print(f"📚 找到現有模型 {old_model_path}，將其作為 Teacher。")
        shutil.copyfile(old_model_path, teacher_model_path)

        trainer = ModelTrainer(
            config,
            data_manager,
            use_incremental=True,
            replay_ratio=0.7,
            lambda_kd=1.0,
            kd_temperature=3.0,
            head_warmup_epochs=3,
            new_animators=weak_animators,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor
        )
        
        model, history = trainer.train_model()

        print("\n🏁 弱類別微調完成!")
        print("   - 檢查輸出中的舊/新類準確率（新類即您的弱類別）。")

if __name__ == '__main__':
    main()
