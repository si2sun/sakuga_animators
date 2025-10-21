from test_modelv3 import AnimatorPredictor
import os

def main():
    """主測試函數"""
    video_paths = [
        './norio_matsumoto_173132_000.mp4',
        './hiroyuki_yamashita_221837_001.mp4',
        './hiroyuki_yamashita_221837_003.mp4',
        './龜.mp4',
        './无标题视频——使用Clipchamp制作 (1).mp4',
        './yoshinori_kanada_28416_001.mp4',
        './yoshinori_kanada_271984_003.mp4',
        './yoshinori_kanada_110059_001.webm'
    ]
    
    # 初始化預測器
    predictor = AnimatorPredictor('./models/best_model.pth')
    
    # 批量預測
    all_results = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            result = predictor.predict(video_path)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️  檔案不存在: {video_path}")
    
    # 匯總統計
    if all_results:
        print("\n" + "="*60)
        print("📊 批量測試匯總")
        print("="*60)
        print(f"測試影片數: {len(all_results)}")
        avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
        print(f"平均置信度: {avg_confidence*100:.2f}%")
        
        # ⭐ 顯示特徵維度資訊
        if all_results[0].get('use_edge_features'):
            print(f"特徵維度: {all_results[0]['feature_dim']} (含 {predictor.edge_dim} 維邊緣特徵)")
        else:
            print(f"特徵維度: {all_results[0]['feature_dim']} (無邊緣特徵)")
        
        if predictor.is_hybrid and 'hybrid_analysis' in all_results[0]:
            print("\n🔍 混合模型整體分析:")
            tcn_correct = sum(1 for r in all_results if r['hybrid_analysis']['tcn_prediction'] == r['predicted_animator'])
            matching_correct = sum(1 for r in all_results if r['hybrid_analysis']['matching_prediction'] == r['predicted_animator'])
            print(f"  TCN分支與最終結果一致: {tcn_correct}/{len(all_results)}")
            print(f"  特徵比對與最終結果一致: {matching_correct}/{len(all_results)}")
        
        print("="*60)

if __name__ == '__main__':
    main()