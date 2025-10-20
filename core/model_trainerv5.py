import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from configs.base_config import BaseConfig

# 焦點損失函數
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None and isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(self.alpha, device=inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 知識蒸餾損失
class KDLoss(nn.Module):
    def __init__(self, temperature=2.0, lambda_kd=1.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.lambda_kd = lambda_kd

    def forward(self, student_logits, teacher_logits, labels=None, old_classes=None):
        if old_classes is not None:
            student_old = student_logits[:, old_classes]
            teacher_old = teacher_logits[:, old_classes]
        else:
            student_old = student_logits
            teacher_old = teacher_logits

        kd_loss = F.kl_div(
            F.log_softmax(student_old / self.temperature, dim=1),
            F.softmax(teacher_old / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return self.lambda_kd * kd_loss

# TCN + BiLSTM 時間卷積網路
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.3, lstm_hidden_dim=256, edge_dim=64):
        super(TemporalConvNet, self).__init__()
        self.edge_dim = edge_dim
        self.core_dim = input_dim - edge_dim
        
        self.layers = nn.ModuleList()
        dilations = [1, 2, 4, 8]
        
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else dilations[-1]
            layer = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, 
                         dilation=dilation, padding=dilation),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
        
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        x_tcn = x.transpose(1, 2)
        for layer in self.layers:
            residual = x_tcn
            x_tcn = layer(x_tcn)
            if x_tcn.shape == residual.shape:
                x_tcn = x_tcn + residual
        
        x_lstm = x_tcn.transpose(1, 2)
        lstm_out, _ = self.bilstm(x_lstm)
        global_feature = lstm_out.mean(dim=1)
        output = self.classifier(global_feature)
        return output, global_feature

# 特徵比對模塊
class FeatureMatchingModule(nn.Module):
    def __init__(self, feature_dim=1344, num_classes=10, edge_dim=64):
        super(FeatureMatchingModule, self).__init__()
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.core_dim = feature_dim - edge_dim
        self.num_classes = num_classes
        
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )
        nn.init.xavier_uniform_(self.class_prototypes)
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        self.core_projection = nn.Sequential(
            nn.Linear(self.core_dim, self.core_dim),
            nn.ReLU(),
            nn.Linear(self.core_dim, self.core_dim)
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(self.edge_dim, self.edge_dim),
            nn.ReLU(),
            nn.Linear(self.edge_dim, self.edge_dim)
        )
    
    def forward(self, features):
        batch_size, seq_len, feature_dim = features.shape
        video_feature = features.mean(dim=1)
        
        core_features = video_feature[:, :self.core_dim]
        edge_features = video_feature[:, self.core_dim:]
        
        projected_core = self.core_projection(core_features)
        projected_edge = self.edge_projection(edge_features)
        projected_feature = torch.cat([projected_core, projected_edge], dim=1)
        
        projected_feature = F.normalize(projected_feature, p=2, dim=1)
        prototypes = F.normalize(self.class_prototypes, p=2, dim=1)
        
        similarity = torch.mm(projected_feature, prototypes.t())
        similarity = similarity / self.temperature
        
        return similarity, projected_feature

# 混合模型
class HybridFeatureTCNModel(nn.Module):
    def __init__(self, config, num_classes, edge_dim=64):
        super(HybridFeatureTCNModel, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.edge_dim = edge_dim
        self.core_dim = 1280
        
        self.tcn = TemporalConvNet(
            input_dim=self.core_dim + self.edge_dim,
            output_dim=num_classes,
            num_layers=3,
            dropout=0.3,
            lstm_hidden_dim=config.lstm_hidden_dim,
            edge_dim=self.edge_dim
        )
        
        self.feature_matching = FeatureMatchingModule(
            feature_dim=self.core_dim + self.edge_dim,
            num_classes=num_classes,
            edge_dim=self.edge_dim
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor([0.6, 0.4]))
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + (self.core_dim + self.edge_dim), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_features=False):
        tcn_logits, tcn_features = self.tcn(x)
        matching_logits, matching_features = self.feature_matching(x)
        
        weights = F.softmax(self.fusion_weight, dim=0)
        weighted_logits = weights[0] * tcn_logits + weights[1] * matching_logits
        
        combined_features = torch.cat([tcn_features, matching_features], dim=1)
        fusion_logits = self.fusion_classifier(combined_features)
        
        final_logits = 0.5 * weighted_logits + 0.5 * fusion_logits
        
        if return_features:
            return final_logits, {
                'tcn_logits': tcn_logits,
                'matching_logits': matching_logits,
                'tcn_features': tcn_features,
                'matching_features': matching_features,
                'fusion_weights': weights
            }
        
        return final_logits

class FeatureDataset(Dataset):
    def __init__(self, feature_samples, label_to_idx, mode='train'):
        self.feature_samples = feature_samples
        self.label_to_idx = label_to_idx
        self.mode = mode
    
    def __len__(self):
        return len(self.feature_samples)
    
    def __getitem__(self, idx):
        feature_path, label = self.feature_samples[idx]
        features = np.load(feature_path, allow_pickle=True)
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        label_idx = self.label_to_idx[label]
        return torch.FloatTensor(features), torch.tensor(label_idx)

def collate_fn_dynamic(batch):
    features_list, labels_list = zip(*batch)
    max_len = max(f.shape[0] for f in features_list)
    
    padded_features = []
    for features in features_list:
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        seq_len = features.shape[0]
        if seq_len < max_len:
            # ⭐⭐⭐ 恢复为使用最后一帧填充 (安全!) ⭐⭐⭐
            last_frame = features[-1:]
            padding = last_frame.repeat(max_len - seq_len, 1)
            padded = torch.cat([features, padding], dim=0)
        else:
            padded = features
        padded_features.append(padded)
    
    features_batch = torch.stack(padded_features)
    labels_list = torch.tensor(labels_list, dtype=torch.long) # 确保 labels 是 long 类型
    
    return features_batch, labels_list

class ModelTrainer:
    def __init__(self, config, data_manager, use_incremental=False,use_kd=False, replay_ratio=0.0, 
                 lambda_kd=1.0, kd_temperature=2.0, head_warmup_epochs=0, 
                 new_animators=None, use_kd_only=False, early_stopping_patience=5, 
                 early_stopping_min_delta=0.001, early_stopping_monitor='test_acc'):
        self.config = config
        self.data_manager = data_manager
        self.use_incremental = use_incremental
        self.use_kd_only = use_kd_only
        self.replay_ratio = replay_ratio
        self.lambda_kd = lambda_kd
        self.kd_temperature = kd_temperature
        self.head_warmup_epochs = head_warmup_epochs
        self.new_animators = new_animators
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Early Stopping 參數
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.best_metric = -float('inf') if early_stopping_monitor == 'test_acc' else float('inf')
        self.epochs_without_improvement = 0
        
        # 設置新舊類別
        self.old_animators = None
        self.old_num_classes = 0
        if self.new_animators:
            all_animators = set(self.data_manager.label_to_idx.keys())
            self.old_animators = sorted(list(all_animators - set(self.new_animators)))
            self.old_num_classes = len(self.old_animators)
        
        # 新增：為弱類別設置 FocalLoss 的 alpha 權重
        self.class_weights = None
        if self.new_animators:
            num_classes = self.data_manager.get_num_classes()
            self.class_weights = [1.0] * num_classes  # 預設權重 1.0
            for animator in self.new_animators:
                idx = self.data_manager.label_to_idx[animator]
                self.class_weights[idx] = 2.0  # 弱類別權重 x2
            print(f"⚖️ 類別權重: {self.class_weights}")
        
        self.teacher_model = None
        if self.use_incremental or self.use_kd_only:
            teacher_path = os.path.join(self.config.model_dir, 'teacher_model.pth')
            if os.path.exists(teacher_path):
                self.load_teacher_model(teacher_path)
            else:
                print("⚠️ 未找到 teacher 模型，KD 將自動禁用")
                if self.use_kd_only:
                    self.use_kd_only = False
                if self.use_incremental:
                    print("   ⚠️ Incremental 模式將僅使用 Replay，不使用 KD")
        
    def load_teacher_model(self, teacher_path):
        checkpoint = torch.load(teacher_path, map_location=self.device)
        teacher_num_classes = checkpoint['num_classes']
        self.teacher_model = HybridFeatureTCNModel(
            self.config, teacher_num_classes, edge_dim=checkpoint.get('edge_dim', 64)
        ).to(self.device)
        self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher_model.eval()
        print(f"✅ 載入 Teacher 模型: {teacher_path}, 類別數: {teacher_num_classes}")

    def initialize_model(self, num_classes, load_pretrained=False):
        """
        初始化模型
        
        Args:
            num_classes: 當前的類別數（可能比預訓練模型多）
            load_pretrained: 是否載入預訓練權重
        """
        model = HybridFeatureTCNModel(self.config, num_classes).to(self.device)
        
        if load_pretrained and (self.use_incremental or self.use_kd_only):
            pretrained_path = os.path.join(self.config.model_dir, 'best_model.pth')
            if os.path.exists(pretrained_path):
                print(f"🔄 載入預訓練模型: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                
                # 獲取舊模型的類別數
                old_num_classes = checkpoint.get('num_classes', self.old_num_classes)
                print(f"   舊模型類別數: {old_num_classes}, 新模型類別數: {num_classes}")
                
                if old_num_classes == num_classes:
                    # 類別數相同，直接載入
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"   ✅ 完整載入模型權重")
                else:
                    # 類別數不同，只載入兼容的層（跳過分類頭）
                    model_dict = model.state_dict()
                    pretrained_dict = checkpoint['model_state_dict']
                    
                    # 需要跳過的層（包含類別數的層）
                    skip_keys = [
                        'tcn.classifier.3.weight',           # TCN 最後一層
                        'tcn.classifier.3.bias',
                        'feature_matching.class_prototypes', # 原型向量
                        'fusion_classifier.3.weight',        # Fusion 最後一層
                        'fusion_classifier.3.bias'
                    ]
                    
                    # 過濾掉不兼容的層
                    compatible_dict = {}
                    skipped_keys = []
                    
                    for k, v in pretrained_dict.items():
                        if k in skip_keys:
                            skipped_keys.append(k)
                            continue
                        
                        # 檢查形狀是否匹配
                        if k in model_dict and v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                        else:
                            skipped_keys.append(k)
                    
                    # 載入兼容的權重
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict)
                    
                    print(f"   ✅ 載入 {len(compatible_dict)}/{len(pretrained_dict)} 層")
                    print(f"   ⚠️  跳過 {len(skipped_keys)} 個不兼容層（將隨機初始化）:")
                    for key in skipped_keys[:5]:  # 只顯示前5個
                        print(f"      - {key}")
                    if len(skipped_keys) > 5:
                        print(f"      ... 還有 {len(skipped_keys) - 5} 個")
                    
                    # 特別處理：擴展 class_prototypes（如果需要）
                    if 'feature_matching.class_prototypes' in pretrained_dict:
                        old_prototypes = pretrained_dict['feature_matching.class_prototypes']
                        old_classes, feature_dim = old_prototypes.shape
                        
                        if old_classes < num_classes:
                            # 保留舊類別的原型，新類別隨機初始化
                            new_prototypes = model.feature_matching.class_prototypes.data
                            new_prototypes[:old_classes] = old_prototypes
                            print(f"   🔄 擴展 class_prototypes: {old_classes} → {num_classes}")
            else:
                print(f"   ⚠️  未找到預訓練模型: {pretrained_path}")
        
        criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
        return model, criterion

    def create_data_loaders(self):
        train_samples, test_samples = self.data_manager.get_fixed_split_samples()
        
        if self.use_incremental:
            if self.replay_ratio > 0 and self.old_animators:
                old_samples = [(path, label) for path, label in train_samples 
                              if label in self.old_animators]
                num_old_samples = len(old_samples)
                num_replay = int(num_old_samples * self.replay_ratio)
                np.random.seed(42)
                replay_indices = np.random.choice(num_old_samples, num_replay, replace=False)
                replay_samples = [old_samples[i] for i in replay_indices]
                
                new_samples = [(path, label) for path, label in train_samples 
                              if label in self.new_animators]
                train_samples = replay_samples + new_samples
                print(f"📊 Replay: 使用 {len(replay_samples)} 個舊樣本 + {len(new_samples)} 個新樣本")

        train_dataset = FeatureDataset(train_samples, self.data_manager.label_to_idx, mode='train')
        test_dataset = FeatureDataset(test_samples, self.data_manager.label_to_idx, mode='test')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_dynamic,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=0
        )
        return train_loader, test_loader

    def train_epoch(self, model, data_loader, optimizer, criterion, stage='joint'):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        kd_loss_fn = KDLoss(temperature=self.kd_temperature, lambda_kd=self.lambda_kd) if self.teacher_model else None

        for features, labels in tqdm(data_loader, desc=f"訓練 ({stage})"):
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            if self.teacher_model and (self.use_incremental or self.use_kd_only) and kd_loss_fn:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(features)
                teacher_padded = torch.zeros_like(outputs)
                if self.old_num_classes > 0:
                    # 修改點：只取教師模型的前 old_num_classes 個 logits
                    teacher_outputs_old = teacher_outputs[:, :self.old_num_classes]
                    teacher_padded[:, :self.old_num_classes] = teacher_outputs_old
                kd_loss = kd_loss_fn(outputs, teacher_padded, labels, 
                                   old_classes=list(range(self.old_num_classes)))
                loss += kd_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(self, model, data_loader, criterion, monitor_old_classes=False):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        old_correct = 0
        old_total = 0
        new_correct = 0
        new_total = 0
        
        confusion_matrix = {}
        
        with torch.no_grad():
            for features, labels in tqdm(data_loader, desc="評估"):
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if monitor_old_classes and self.old_animators:
                    old_indices = [self.data_manager.label_to_idx[anim] for anim in self.old_animators]
                    new_indices = [self.data_manager.label_to_idx[anim] for anim in self.new_animators]
                    
                    for i in range(labels.size(0)):
                        label = labels[i].item()
                        pred = predicted[i].item()
                        if label in old_indices:
                            old_total += 1
                            if label == pred:
                                old_correct += 1
                        if label in new_indices:
                            new_total += 1
                            if label == pred:
                                new_correct += 1
                        
                        label_key = self.data_manager.idx_to_label[label]
                        pred_key = self.data_manager.idx_to_label[pred]
                        confusion_matrix[(label_key, pred_key)] = confusion_matrix.get((label_key, pred_key), 0) + 1
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        old_accuracy = 100 * old_correct / old_total if old_total > 0 else 0
        new_accuracy = 100 * new_correct / new_total if new_total > 0 else 0
        
        print("\n混淆矩陣統計:")
        for (true_label, pred_label), count in sorted(confusion_matrix.items()):
            print(f"  {true_label} → {pred_label}: {count} 次")
        
        return avg_loss, accuracy, old_accuracy, new_accuracy

    def train_model(self):
        print("🎯 開始訓練混合模型: EfficientNet-V2-S + 邊緣特徵 + TCN + BiLSTM...")
        
        train_loader, test_loader = self.create_data_loaders()
        num_classes = self.data_manager.get_num_classes()
        
        print(f"🎨 識別 {num_classes} 個原畫師: {list(self.data_manager.label_to_idx.keys())}")
        
        model, criterion = self.initialize_model(num_classes, load_pretrained=True)
        
        print("\n🔧 設置差分學習率優化器...")
        
        temporal_body_params = [
            p for name, p in model.named_parameters() 
            if 'tcn.layers' in name or 'tcn.bilstm' in name
        ]
        
        head_params = [
            p for name, p in model.named_parameters() 
            if 'tcn.classifier' in name or 'feature_matching' in name or 'fusion_classifier' in name
        ]
        
        all_param_ids = set(id(p) for p in model.parameters())
        body_param_ids = set(id(p) for p in temporal_body_params)
        head_param_ids = set(id(p) for p in head_params)
        other_param_ids = all_param_ids - body_param_ids - head_param_ids
        other_params = [p for p in model.parameters() if id(p) in other_param_ids]
        if other_params:
            print(f"   - 注意: 檢測到 {len(other_params)} 個額外參數 (如 fusion_weight), 將使用頭部學習率")
            head_params.extend(other_params)

        is_incremental_mode = self.use_incremental or self.use_kd_only
        
        lr_body = 1e-6 if is_incremental_mode else self.config.learning_rate / 100
        lr_heads = 1e-4 if is_incremental_mode else self.config.learning_rate

        optimizer = optim.AdamW([
            {'params': temporal_body_params, 'lr': lr_body},
            {'params': head_params, 'lr': lr_heads}
        ], weight_decay=0.01)

        print(f"   - 模型主體 (TCN/BiLSTM) 學習率: {lr_body}")
        print(f"   - 分類頭 (Classifiers/Matching) 學習率: {lr_heads}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.num_epochs
        )
        
        best_accuracy = 0
        best_loss = float('inf')
        training_history = []
        total_epochs = self.config.num_epochs
        warmup_epochs = self.head_warmup_epochs if (self.use_incremental or self.use_kd_only) else 0
        
        for epoch in range(total_epochs):
            current_stage = 'head_warmup' if epoch < warmup_epochs else 'joint'
            print(f"\n📅 Epoch {epoch+1}/{total_epochs} (當前階段: {current_stage})")
            
            if current_stage == 'head_warmup':
                for g in optimizer.param_groups:
                    if len(g['params']) == len(temporal_body_params):
                         g['lr'] = 0
            elif current_stage == 'joint' and epoch == warmup_epochs:
                print(f"   - 進入 Joint 訓練，恢復主體學習率至 {lr_body}")
                for g in optimizer.param_groups:
                    if len(g['params']) == len(temporal_body_params):
                         g['lr'] = lr_body
            
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, stage=current_stage)
            test_loss, test_acc, old_acc, new_acc = self.evaluate(model, test_loader, criterion, monitor_old_classes=(self.use_incremental or self.use_kd_only))
            
            scheduler.step()
            
            print(f"📊 訓練損失: {train_loss:.4f}, 準確率: {train_acc:.1f}%")
            print(f"🧪 測試損失: {test_loss:.4f}, 準確率: {test_acc:.1f}%")
            
            if (self.use_incremental or self.use_kd_only) and old_acc > 0:
                print(f"   🛡️ 舊類: {old_acc:.1f}%", end="")
                if new_acc > 0:
                    print(f" | 🆕 新類: {new_acc:.1f}%")
                else:
                    print()
                
                if current_stage == 'joint' and best_accuracy > 0 and old_acc < best_accuracy * 0.95:
                    print("⚠️ 舊類準確率下降明顯，考慮增加 replay_ratio 或 lambda_kd")
            
            # Early Stopping 邏輯
            current_metric = test_acc if self.early_stopping_monitor == 'test_acc' else test_loss
            is_improved = False
            
            if self.early_stopping_monitor == 'test_acc':
                if current_metric >= self.best_metric + self.early_stopping_min_delta:
                    is_improved = True
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            else:  # test_loss
                if current_metric <= self.best_metric - self.early_stopping_min_delta:
                    is_improved = True
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            
            if is_improved and test_acc > best_accuracy:
                best_accuracy = test_acc
                best_loss = test_loss
                save_as_teacher = self.use_incremental or self.use_kd_only
                self.save_model(model, num_classes, epoch, test_acc, save_as_teacher=save_as_teacher)
                print(f"💾 保存最佳模型 (準確率: {test_acc:.1f}%)")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'old_acc': old_acc if (self.use_incremental or self.use_kd_only) else None,
                'new_acc': new_acc if (self.use_incremental or self.use_kd_only) else None,
                'learning_rate_body': optimizer.param_groups[0]['lr'],
                'learning_rate_head': optimizer.param_groups[1]['lr'],
                'stage': current_stage
            })
            
            # 檢查 Early Stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n🛑 Early Stopping 觸發: 在 {self.early_stopping_patience} 個 epoch 內未改善 {self.early_stopping_monitor}")
                print(f"   最佳 {self.early_stopping_monitor}: {self.best_metric:.4f}")
                break
        
        self.save_exemplar_buffer(model)
        
        print(f"\n🏆 訓練完成! 最佳測試準確率: {best_accuracy:.1f}%")
        return model, training_history
    
    def save_exemplar_buffer(self, model):
        print("\n💾 生成 exemplar_buffer.json...")
        
        all_train_samples, _ = self.data_manager.get_fixed_split_samples()
        
        buffer_data = {}
        for feature_path, animator in all_train_samples:
            animator_idx = self.data_manager.label_to_idx[animator]
            
            if animator_idx not in buffer_data:
                buffer_data[animator_idx] = []
            
            buffer_data[animator_idx].append(feature_path)
        
        buffer_path = os.path.join(self.config.model_dir, 'exemplar_buffer.json')
        with open(buffer_path, 'w', encoding='utf-8') as f:
            json.dump(buffer_data, f, ensure_ascii=False, indent=2)
        
        total_samples = sum(len(paths) for paths in buffer_data.values())
        print(f"✅ exemplar_buffer.json 已生成:")
        print(f"   總樣本數: {total_samples}")
        print(f"   原畫師數: {len(buffer_data)}")
        for idx, paths in buffer_data.items():
            animator = self.data_manager.idx_to_label[idx]
            print(f"   {animator}: {len(paths)} 個樣本")
        print(f"   保存路徑: {buffer_path}")
    
    def save_model(self, model, num_classes, epoch, accuracy, save_as_teacher=False):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'epoch': epoch,
            'accuracy': accuracy,
            'label_mapping': {
                'label_to_idx': self.data_manager.label_to_idx,
                'idx_to_label': self.data_manager.idx_to_label
            },
            'config': self.config.__dict__,
            'model_type': 'hybrid_edge',
            'edge_dim': 64,
            'old_num_classes': self.old_num_classes,
            'new_animators': self.new_animators,
            'old_animators': self.old_animators
        }
        
        # 保存為 train_model.pth（訓練輸出）
        train_model_path = os.path.join(self.config.model_dir, 'best_model.pth')
        torch.save(checkpoint, train_model_path)
        print(f"   💾 保存訓練模型: {train_model_path}")
        
        # ⭐ 如果是增量學習，自動更新 best_model.pth
        # if save_as_teacher:
        #     best_model_path = os.path.join(self.config.model_dir, 'best_model.pth')
        #     torch.save(checkpoint, best_model_path)
        # print(f"   ✅ 同步更新 best_model.pth（供下次增量使用）")
        # torch.save(checkpoint, os.path.join(self.config.model_dir, 'latest_model.pth'))

        
        # 無論如何都保存 best 和 latest
        # torch.save(checkpoint, os.path.join(self.config.model_dir, 'best_model.pth'))
        # torch.save(checkpoint, os.path.join(self.config.model_dir, 'latest_model.pth'))