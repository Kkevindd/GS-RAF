#!/bin/bash

# =============================================================================
# 音频场评估脚本 - 支持70维局部特征和全局特征
# =============================================================================
# 用法：
#   ./evaluate.sh <model_path> <raf_data> [local|global] [其他选项]
#
# 示例：
#   ./evaluate.sh ./output/my_model /path/to/raf local
#   ./evaluate.sh ./output/my_model/audio_ckpts/audio_iter_200.pth /path/to/raf local
#   ./evaluate.sh ./output/my_model /path/to/raf global --max_samples 50
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示使用帮助
show_help() {
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${GREEN}音频场评估脚本 - 使用说明${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo "用法："
    echo "  $0 <model_path> <raf_data> [feature_type] [其他选项]"
    echo ""
    echo "参数："
    echo "  model_path      模型路径（可以是目录或checkpoint文件）"
    echo "  raf_data        RAF数据集路径"
    echo "  feature_type    特征类型: 'local' (70维局部) 或 'global' (全局), 默认: local"
    echo ""
    echo "可选参数："
    echo "  --checkpoint NAME       指定checkpoint文件名 (如: audio_iter_200.pth)"
    echo "  --max_samples N         最大评估样本数"
    echo "  --max_vis_samples N     最大可视化样本数 (默认: 10)"
    echo "  --output_dir PATH       输出目录"
    echo "  --no_visualizations     不保存可视化图像"
    echo ""
    echo "示例："
    echo "  # 使用局部特征评估（默认）"
    echo "  $0 ./output/my_model /path/to/raf local"
    echo ""
    echo "  # 直接传入checkpoint文件"
    echo "  $0 ./output/my_model/audio_ckpts/audio_iter_200.pth /path/to/raf local"
    echo ""
    echo "  # 使用全局特征评估"
    echo "  $0 ./output/my_model /path/to/raf global"
    echo ""
    echo "  # 限制评估样本数"
    echo "  $0 ./output/my_model /path/to/raf local --max_samples 50"
    echo ""
    echo "  # 指定checkpoint"
    echo "  $0 ./output/my_model /path/to/raf local --checkpoint audio_iter_500.pth"
    echo ""
    echo -e "${BLUE}=====================================================================${NC}"
}

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${RED}错误: 参数不足${NC}"
    echo ""
    show_help
    exit 1
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

# 读取必需参数
MODEL_PATH="$1"
RAF_DATA="$2"
FEATURE_TYPE="${3:-local}"  # 默认使用local

# 检查特征类型
if [ "$FEATURE_TYPE" != "local" ] && [ "$FEATURE_TYPE" != "global" ] && [ "$FEATURE_TYPE" != "--"* ]; then
    echo -e "${RED}错误: 特征类型必须是 'local' 或 'global', 当前: $FEATURE_TYPE${NC}"
    exit 1
fi

# 如果第三个参数是选项（以--开头），则默认使用local
if [[ "$FEATURE_TYPE" == --* ]]; then
    FEATURE_TYPE="local"
    shift 2  # 只移除前两个参数
else
    shift 3  # 移除前三个参数
fi

# 剩余的参数传递给Python脚本
EXTRA_ARGS="$@"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}音频场评估脚本${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# 智能处理模型路径
ACTUAL_MODEL_PATH="$MODEL_PATH"
CHECKPOINT_ARG=""

if [ -f "$MODEL_PATH" ] && [[ "$MODEL_PATH" == *.pth ]]; then
    # 传入的是checkpoint文件
    echo -e "${YELLOW}📌 检测到checkpoint文件路径${NC}"
    CHECKPOINT_FILE=$(basename "$MODEL_PATH")
    AUDIO_CKPTS_DIR=$(dirname "$MODEL_PATH")
    ACTUAL_MODEL_PATH=$(dirname "$AUDIO_CKPTS_DIR")
    
    # 检查是否已经通过--checkpoint指定了
    if [[ ! "$EXTRA_ARGS" =~ "--checkpoint" ]]; then
        CHECKPOINT_ARG="--checkpoint $CHECKPOINT_FILE"
    fi
    
    echo -e "   原始路径: ${BLUE}$MODEL_PATH${NC}"
    echo -e "   模型目录: ${GREEN}$ACTUAL_MODEL_PATH${NC}"
    echo -e "   Checkpoint: ${GREEN}$CHECKPOINT_FILE${NC}"
    echo ""
elif [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓ 模型目录: $MODEL_PATH${NC}"
    echo ""
else
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 检查RAF数据路径
if [ ! -d "$RAF_DATA" ]; then
    echo -e "${RED}错误: RAF数据路径不存在: $RAF_DATA${NC}"
    exit 1
fi
echo -e "${GREEN}✓ RAF数据: $RAF_DATA${NC}"
echo ""

# 设置特征类型参数
if [ "$FEATURE_TYPE" == "local" ]; then
    FEATURE_ARG="--use_local_features"
    echo -e "${GREEN}✓ 特征类型: 局部特征 (70维)${NC}"
else
    FEATURE_ARG="--use_global_features"
    echo -e "${YELLOW}✓ 特征类型: 全局特征${NC}"
fi
echo ""

# 构建完整命令
CMD="python test_audio_metrics.py \
  --model_path \"$ACTUAL_MODEL_PATH\" \
  $CHECKPOINT_ARG \
  --raf_data \"$RAF_DATA\" \
  $FEATURE_ARG \
  $EXTRA_ARGS"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}执行命令:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# 执行评估
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${GREEN}✓ 评估完成！${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    
    # 显示结果文件位置
    RESULTS_DIR="$ACTUAL_MODEL_PATH/audio_neraf_results"
    if [ -d "$RESULTS_DIR" ]; then
        echo ""
        echo -e "${GREEN}结果保存在:${NC}"
        echo -e "  ${BLUE}$RESULTS_DIR${NC}"
        echo ""
        echo -e "${GREEN}可视化图像 (如果生成):${NC}"
        echo -e "  ${BLUE}$ACTUAL_MODEL_PATH/visualizations${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ 评估失败！${NC}"
    exit 1
fi

