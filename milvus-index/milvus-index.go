package main

import (
	"context"
	"log"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	//milvusAddr := "10.50.56.243:19530"
	milvusAddr := "milvus-test.com:443"
	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:       milvusAddr,
		Username:      "test",
		Password:      "123456",
		DBName:        "test_db",
		EnableTLSAuth: true,
	})
	if err != nil {
		log.Fatalf("连接 Milvus 失败: %v", err)
	}
	log.Println("成功连接到 Milvus")
	defer func() {
		if err := cli.Close(ctx); err != nil {
			log.Printf("关闭 Milvus 连接失败: %v", err)
		} else {
			log.Println("Milvus 连接已成功关闭")
		}
	}()

	// 为不同字段定义索引
	// 为不同字段定义索引并设置自定义名称
	indexConfig := map[string]struct {
		index.Index
		customName string
	}{
		"vector": {
			Index:      index.NewHNSWIndex(entity.COSINE, 32, 128),
			customName: "vector_hnsw_idx", // 自定义索引名称
		},
		"vec_2": {
			Index:      index.NewIvfFlatIndex(entity.IP, 32),
			customName: "vec2_ivfflat_idx", // 自定义索引名称
		},
		"vec_3": {
			Index:      index.NewIvfFlatIndex(entity.IP, 32),
			customName: "vec3_ivfflat_idx", // 自定义索引名称
		},
	}

	collectionName := "test3"
	for fieldName, config := range indexConfig {
		// 创建前检查索引是否已存在
		indexinfo, err := cli.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(collectionName, config.customName))
		if err != nil {
			log.Printf("查询字段 %s 不存在索引: %v", fieldName, err)
		} else if indexinfo.Index.Name() != "" {
			log.Printf("字段 %s 已存在索引，跳过创建", fieldName)
			continue
		}

		// 创建新索引
		log.Printf("为集合 %s 的字段 %s 创建索引", collectionName, fieldName)
		// 创建索引选项并设置自定义名称
		indexOption := milvusclient.NewCreateIndexOption(collectionName, fieldName, config.Index)
		indexOption.WithIndexName(config.customName) // 设置自定义索引名称
		// 创建索引
		indexTask, err := cli.CreateIndex(ctx, indexOption)
		if err != nil {
			log.Printf("创建字段 %s 的索引失败: %v", fieldName, err)
			continue
		}

		// 仅当任务创建成功时才等待完成
		log.Printf("等待字段 %s 的索引创建完成", fieldName)
		if err := indexTask.Await(ctx); err != nil {
			log.Printf("字段 %s 的索引创建失败: %v", fieldName, err)
		} else {
			log.Printf("字段 %s 的索引创建成功", fieldName)
		}
	}

	log.Println("索引创建流程已完成")
}
