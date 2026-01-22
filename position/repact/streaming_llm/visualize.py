import matplotlib.pyplot as plt
import torch

def visualize_selected_attentions(attentions, layer_indices=[0, 1, 16, 30, 31], head=0, input_tokens=None):
    """
    특정 레이어의 Attention Score를 시각화하는 함수
    
    Args:
        outputs: model(input_ids, output_attentions=True) 결과 객체
        layer_indices (list): 시각화할 레이어 인덱스 리스트
        head (int): 시각화할 Attention head 인덱스 (기본값 0)
        input_tokens (list[str], optional): 시각화할 토큰 문자열 리스트 (tick label용)
    """
    
    
    # Plot 설정
    num_layers = len(layer_indices)
    plt.figure(figsize=(16, 3 * num_layers))
    
    for i, layer_idx in enumerate(layer_indices):
        attn = attentions[layer_idx][0, head].detach().cpu()  # (seq_len, seq_len)

        plt.subplot(num_layers, 1, i + 1)
        plt.imshow(attn, cmap='viridis')
        plt.title(f"Layer {layer_idx} - Head {head}")
        plt.colorbar(label='Attention Score')
        
        if input_tokens is not None:
            seq_len = len(input_tokens)
            plt.xticks(range(seq_len), input_tokens, rotation=90)
            plt.yticks(range(seq_len), input_tokens)
    
    plt.tight_layout()
    plt.savefig('dec_attn_after_comp_posshift.png', dpi=300, format='png')
    exit()

