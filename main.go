package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Config 配置结构
type Config struct {
	Source struct {
		Host     string `json:"host"`
		Port     int    `json:"port"`
		Username string `json:"username"`
		Password string `json:"password"`
		Database string `json:"database"`
	} `json:"source"`
	Target struct {
		Host     string `json:"host"`
		Port     int    `json:"port"`
		Username string `json:"username"`
		Password string `json:"password"`
		Database string `json:"database"`
	} `json:"target"`
	Collections []string `json:"collections"`
	BatchSize   int      `json:"batch_size"`
	Parallel    int      `json:"parallel_workers"`
}

// MilvusMigrator 迁移器
type MilvusMigrator struct {
	sourceClient *milvusclient.Client
	targetClient *milvusclient.Client
	config       *Config
}

// createClientForDatabase 为指定数据库创建客户端连接
func createClientForDatabase(host string, port int, username, password, database string) (*milvusclient.Client, error) {
	log.Printf("[DEBUG] createClientForDatabase 输入参数: host='%s', port=%d, database='%s'", host, port, database)

	// 解析host字段，支持http://、https://、tcps://等协议前缀
	var addr string
	var useTLS bool
	originalHost := host // 保存原始host用于日志

	// 检查host是否包含协议前缀
	if strings.HasPrefix(host, "http://") {
		// HTTP协议，移除前缀
		host = strings.TrimPrefix(host, "http://")
		useTLS = false
	} else if strings.HasPrefix(host, "https://") {
		// HTTPS协议，移除前缀
		host = strings.TrimPrefix(host, "https://")
		useTLS = true
	} else if strings.HasPrefix(host, "tcps://") {
		// TCPS协议，移除前缀
		host = strings.TrimPrefix(host, "tcps://")
		useTLS = true
	} else if strings.HasPrefix(host, "tcp://") {
		// TCP协议，移除前缀
		host = strings.TrimPrefix(host, "tcp://")
		useTLS = false
	} else {
		// 没有协议前缀，默认不使用TLS
		useTLS = false
	}

	// 构建连接地址 - 不需要协议前缀
	addr = fmt.Sprintf("%s:%d", host, port)

	log.Printf("[DEBUG] 为数据库 %s 创建连接: %s (原始配置: %s, TLS: %v)", database, addr, originalHost, useTLS)

	config := &milvusclient.ClientConfig{
		Address:       addr,
		Username:      username,
		Password:      password,
		EnableTLSAuth: useTLS, // 使用EnableTLSAuth字段启用TLS
	}

	// 添加连接超时
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Printf("[DEBUG] 尝试连接到 %s...", addr)
	client, err := milvusclient.New(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("创建数据库 %s 的连接失败: %v", database, err)
	}

	// 如果指定了数据库，使用UseDatabase方法切换
	if database != "" && database != "default" {
		log.Printf("[DEBUG] 切换到数据库: %s", database)
		err = client.UseDatabase(context.Background(), milvusclient.NewUseDatabaseOption(database))
		if err != nil {
			log.Printf("[DEBUG] 警告: 切换到数据库 %s 失败: %v", database, err)
		}
	}

	log.Printf("[DEBUG] 数据库 %s 连接成功", database)
	return client, nil
}

// NewMilvusMigrator 创建新的迁移器
func NewMilvusMigrator(configPath string) (*MilvusMigrator, error) {
	log.Printf("[DEBUG] 开始创建迁移器，配置文件: %s", configPath)

	config, err := loadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("加载配置失败: %v", err)
	}

	// 显示实际读取的配置信息
	log.Printf("[DEBUG] 配置文件内容:")
	log.Printf("[DEBUG]   源数据库: host='%s', port=%d, database='%s'",
		config.Source.Host, config.Source.Port, config.Source.Database)
	log.Printf("[DEBUG]   目标数据库: host='%s', port=%d, database='%s'",
		config.Target.Host, config.Target.Port, config.Target.Database)

	migrator := &MilvusMigrator{config: config}

	// 为源数据库创建连接
	sourceDatabase := config.Source.Database
	if sourceDatabase == "" {
		sourceDatabase = "default"
	}
	log.Printf("[DEBUG] 开始创建源数据库连接...")
	sourceCli, err := createClientForDatabase(
		config.Source.Host,
		config.Source.Port,
		config.Source.Username,
		config.Source.Password,
		sourceDatabase,
	)
	if err != nil {
		return nil, fmt.Errorf("创建源数据库连接失败: %v", err)
	}
	migrator.sourceClient = sourceCli

	// 为目标数据库创建连接
	targetDatabase := config.Target.Database
	if targetDatabase == "" {
		targetDatabase = "default"
	}
	log.Printf("[DEBUG] 开始创建目标数据库连接...")
	targetCli, err := createClientForDatabase(
		config.Target.Host,
		config.Target.Port,
		config.Target.Username,
		config.Target.Password,
		targetDatabase,
	)
	if err != nil {
		return nil, fmt.Errorf("创建目标数据库连接失败: %v", err)
	}
	migrator.targetClient = targetCli

	log.Printf("[DEBUG] 迁移器创建完成")
	return migrator, nil
}

func (m *MilvusMigrator) Close(ctx context.Context) {
	log.Printf("[DEBUG] 开始关闭迁移器连接")
	if m.sourceClient != nil {
		log.Printf("[DEBUG] 关闭源Milvus连接")
		m.sourceClient.Close(ctx)
	}
	if m.targetClient != nil {
		log.Printf("[DEBUG] 关闭目标Milvus连接")
		m.targetClient.Close(ctx)
	}
	log.Printf("[DEBUG] 迁移器连接已关闭")
}

// 迁移主流程（collection/schema/索引/数据）
func (m *MilvusMigrator) MigrateCollection(ctx context.Context, collectionName string) error {
	log.Printf("[DEBUG] 开始迁移collection: %s", collectionName)

	// 1. 获取源collection schema
	schemaColl, err := m.sourceClient.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		return fmt.Errorf("获取源collection schema失败: %v", err)
	}
	schema := schemaColl.Schema
	log.Printf("[DEBUG] 源collection schema获取成功，字段数: %d", len(schema.Fields))

	// 2. 检查目标collection是否存在
	hasCollection, err := m.targetClient.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return fmt.Errorf("检查目标collection失败: %v", err)
	}

	// 3. 创建目标collection（如果不存在）
	if !hasCollection {
		log.Printf("[DEBUG] 创建目标collection: %s", collectionName)
		err = m.targetClient.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema))
		if err != nil {
			return fmt.Errorf("创建目标collection失败: %v", err)
		}
	} else {
		log.Printf("[DEBUG] 目标collection已存在: %s", collectionName)
	}

	// 4. 获取源collection数据总数
	stats, err := m.sourceClient.GetCollectionStats(ctx, milvusclient.NewGetCollectionStatsOption(collectionName))
	if err != nil {
		return fmt.Errorf("获取collection统计信息失败: %v", err)
	}
	rowCount, _ := strconv.ParseInt(stats["row_count"], 10, 64)
	log.Printf("[DEBUG] Collection %s 总记录数: %d", collectionName, rowCount)

	// 5. 分批迁移数据
	err = m.migrateData(ctx, collectionName, collectionName, schema, int(rowCount))
	if err != nil {
		return fmt.Errorf("迁移数据失败: %v", err)
	}

	log.Printf("[DEBUG] Collection %s 迁移完成", collectionName)
	return nil
}

// migrateData 迁移数据
func (m *MilvusMigrator) migrateData(ctx context.Context, sourceCollectionName, targetCollectionName string, schema *entity.Schema, totalRows int) error {
	log.Printf("[DEBUG] 开始迁移数据，总行数: %d", totalRows)

	batchSize := m.config.BatchSize
	if batchSize <= 0 {
		batchSize = 1000
	}
	log.Printf("[DEBUG] 批次大小: %d", batchSize)

	// 获取非 autoID 字段名
	log.Printf("[DEBUG] 分析schema字段，总字段数: %d", len(schema.Fields))
	insertFields := make([]*entity.Field, 0, len(schema.Fields))
	insertFieldNames := make([]string, 0, len(schema.Fields))
	for _, field := range schema.Fields {
		log.Printf("[DEBUG] 字段: %s, AutoID: %v", field.Name, field.AutoID)
		if !field.AutoID {
			insertFields = append(insertFields, field)
			insertFieldNames = append(insertFieldNames, field.Name)
		}
	}
	log.Printf("[DEBUG] 需要插入的字段数: %d, 字段名: %v", len(insertFields), insertFieldNames)

	// 获取目标 collection schema 字段名
	log.Printf("[DEBUG] 获取目标collection schema")
	targetSchemaColl, err := m.targetClient.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(targetCollectionName))
	if err != nil {
		return fmt.Errorf("获取目标collection schema失败: %v", err)
	}
	targetSchema := targetSchemaColl.Schema
	targetFieldNames := make([]string, 0, len(targetSchema.Fields))
	for _, field := range targetSchema.Fields {
		targetFieldNames = append(targetFieldNames, field.Name)
	}
	log.Printf("[DEBUG] 目标collection字段数: %d, 字段名: %v", len(targetSchema.Fields), targetFieldNames)

	log.Printf("[DEBUG] 加载源collection到内存")
	_, err = m.sourceClient.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(sourceCollectionName))
	if err != nil {
		return fmt.Errorf("加载源collection失败: %v", err)
	}
	log.Printf("[DEBUG] 源collection加载成功")

	batches := (totalRows + batchSize - 1) / batchSize
	log.Printf("[DEBUG] 将分 %d 批次迁移数据，每批 %d 条记录", batches, batchSize)

	for i := 0; i < batches; i++ {
		offset := i * batchSize
		limit := batchSize
		if offset+limit > totalRows {
			limit = totalRows - offset
		}

		log.Printf("[DEBUG] 开始迁移批次 %d/%d (记录 %d-%d)", i+1, batches, offset+1, offset+limit)

		log.Printf("[DEBUG] 查询批次数据")
		data, err := m.queryBatch(ctx, sourceCollectionName, insertFields, offset, limit)
		if err != nil {
			return fmt.Errorf("查询批次数据失败: %v", err)
		}
		log.Printf("[DEBUG] 批次数据查询成功，返回列数: %d", len(data))

		log.Printf("[DEBUG] 插入字段名: %v", insertFieldNames)
		log.Printf("[DEBUG] 目标collection字段名: %v", targetFieldNames)
		log.Printf("[DEBUG] 插入数据列数: %d", len(data))
		for idx, col := range data {
			if c, ok := col.(column.Column); ok {
				log.Printf("[DEBUG] 第%d列字段名: %s, 数据长度: %d", idx+1, c.Name(), c.Len())
			}
		}

		log.Printf("[DEBUG] 过滤数据列")
		filteredData := make([]column.Column, 0, len(insertFields))
		for _, wantField := range insertFields {
			for _, col := range data {
				if c, ok := col.(column.Column); ok {
					if c.Name() == wantField.Name {
						filteredData = append(filteredData, c)
						log.Printf("[DEBUG] 找到匹配字段: %s", wantField.Name)
						break
					}
				}
			}
		}
		log.Printf("[DEBUG] 过滤后数据列数: %d", len(filteredData))

		if len(filteredData) > 0 {
			log.Printf("[DEBUG] 开始插入数据到目标collection")
			_, err = m.targetClient.Insert(ctx, milvusclient.NewColumnBasedInsertOption(targetCollectionName, filteredData...))
			if err != nil {
				return fmt.Errorf("插入数据失败: %v", err)
			}
			log.Printf("[DEBUG] 批次 %d 数据插入成功", i+1)
		} else {
			log.Printf("[DEBUG] 批次 %d 没有数据需要插入", i+1)
		}
	}

	log.Printf("[DEBUG] 所有批次数据迁移完成，开始刷新")
	_, err = m.targetClient.Flush(ctx, milvusclient.NewFlushOption(targetCollectionName))
	if err != nil {
		log.Printf("[DEBUG] 警告: 刷新目标collection失败: %v", err)
	} else {
		log.Printf("[DEBUG] 目标collection刷新成功")
	}

	log.Printf("[DEBUG] 数据迁移完成")
	return nil
}

// queryBatch 查询批次数据（只查指定字段）
func (m *MilvusMigrator) queryBatch(ctx context.Context, collectionName string, fields []*entity.Field, offset, limit int) ([]column.Column, error) {
	log.Printf("[DEBUG] 开始查询批次数据，offset: %d, limit: %d", offset, limit)

	outputFields := make([]string, 0, len(fields))
	for _, field := range fields {
		outputFields = append(outputFields, field.Name)
	}
	log.Printf("[DEBUG] 查询字段: %v", outputFields)

	log.Printf("[DEBUG] 执行查询操作")
	results, err := m.sourceClient.Query(
		ctx,
		milvusclient.NewQueryOption(collectionName).WithOutputFields(outputFields...).WithLimit(int(limit)).WithOffset(int(offset)),
	)
	if err != nil {
		log.Printf("[DEBUG] 查询失败: %v", err)
		return nil, err
	}
	log.Printf("[DEBUG] 查询成功，返回字段数: %d", len(results.Fields))

	cols := make([]column.Column, 0, len(results.Fields))
	for _, col := range results.Fields {
		if c, ok := col.(column.Column); ok {
			cols = append(cols, c)
			log.Printf("[DEBUG] 处理列: %s, 长度: %d", c.Name(), c.Len())
		}
	}
	log.Printf("[DEBUG] 查询批次完成，返回列数: %d", len(cols))
	return cols, nil
}

// MigrateAll 迁移所有指定的collections
func (m *MilvusMigrator) MigrateAll(ctx context.Context) error {
	log.Printf("[DEBUG] 开始迁移所有collections")

	collections := m.config.Collections
	if len(collections) == 0 {
		log.Printf("[DEBUG] 未指定collections，获取所有collections")
		allCollections, err := m.sourceClient.ListCollections(ctx, milvusclient.NewListCollectionOption())
		if err != nil {
			return fmt.Errorf("获取collection列表失败: %v", err)
		}
		collections = allCollections
		log.Printf("[DEBUG] 获取到 %d 个collections: %v", len(collections), collections)
	} else {
		log.Printf("[DEBUG] 使用指定的 %d 个collections: %v", len(collections), collections)
	}

	log.Printf("[DEBUG] 开始迁移 %d 个collections", len(collections))

	for i, collectionName := range collections {
		log.Printf("[DEBUG] 开始迁移第 %d/%d 个collection: %s", i+1, len(collections), collectionName)
		err := m.MigrateCollection(ctx, collectionName)
		if err != nil {
			log.Printf("[DEBUG] 迁移collection %s 失败: %v", collectionName, err)
			return err
		}
		log.Printf("[DEBUG] 第 %d/%d 个collection迁移完成: %s", i+1, len(collections), collectionName)
	}

	log.Printf("[DEBUG] 所有collections迁移完成")
	return nil
}

// loadConfig 加载配置文件
func loadConfig(configPath string) (*Config, error) {
	log.Printf("[DEBUG] 开始加载配置文件: %s", configPath)

	file, err := os.Open(configPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var config Config
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&config)
	if err != nil {
		return nil, err
	}

	if config.BatchSize <= 0 {
		config.BatchSize = 1000
	}
	if config.Parallel <= 0 {
		config.Parallel = 1
	}

	log.Printf("[DEBUG] 配置文件加载成功")
	return &config, nil
}

// createSampleConfig 创建示例配置文件
func createSampleConfig(filename string) error {
	config := Config{
		BatchSize: 1000,
		Parallel:  1,
	}

	// 源数据库配置示例 - HTTP连接
	config.Source.Host = "http://10.50.60.243" // HTTP协议
	config.Source.Port = 19530
	config.Source.Username = "" // 如果需要认证
	config.Source.Password = "" // 如果需要认证
	config.Source.Database = "test_db"

	// 目标数据库配置示例 - HTTPS连接
	config.Target.Host = "https://milvus-prod.supcon.com:443" // HTTPS协议，包含端口
	config.Target.Port = 19530                                // 这个端口会被忽略，因为host中已包含端口
	config.Target.Username = "your_username"                  // 如果需要认证
	config.Target.Password = "your_password"                  // 如果需要认证
	config.Target.Database = "test001"

	config.Collections = []string{"test"}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(config)
}

func main() {
	log.Printf("[DEBUG] 程序启动")

	var (
		configPath     = flag.String("config", "config.json", "配置文件路径")
		generateConfig = flag.Bool("generate-config", false, "生成示例配置文件")
		collections    = flag.String("collections", "", "要迁移的collections，用逗号分隔（为空则迁移全部）")
	)
	flag.Parse()

	log.Printf("[DEBUG] 命令行参数: config=%s, generate-config=%v, collections=%s",
		*configPath, *generateConfig, *collections)

	if *generateConfig {
		log.Printf("[DEBUG] 生成配置文件模式")
		err := createSampleConfig(*configPath)
		if err != nil {
			log.Fatalf("生成配置文件失败: %v", err)
		}
		fmt.Printf("示例配置文件已生成: %s\n", *configPath)
		return
	}

	log.Printf("[DEBUG] 创建上下文")
	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	defer cancel()

	log.Printf("[DEBUG] 开始创建迁移器")
	migrator, err := NewMilvusMigrator(*configPath)
	if err != nil {
		log.Fatalf("创建迁移器失败: %v", err)
	}
	defer migrator.Close(ctx)

	if *collections != "" {
		log.Printf("[DEBUG] 解析指定的collections: %s", *collections)
		migrator.config.Collections = strings.Split(*collections, ",")
		for i, coll := range migrator.config.Collections {
			migrator.config.Collections[i] = strings.TrimSpace(coll)
		}
		log.Printf("[DEBUG] 解析后的collections: %v", migrator.config.Collections)
	}

	log.Printf("[DEBUG] 开始执行迁移")
	start := time.Now()
	err = migrator.MigrateAll(ctx)
	if err != nil {
		log.Fatalf("迁移失败: %v", err)
	}

	duration := time.Since(start)
	log.Printf("[DEBUG] 迁移完成，总耗时: %v", duration)
}

