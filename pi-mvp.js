/**
 * Pi-MVP — 单文件 AI Agent Toolkit (仅依赖 fs/path/os)
 *
 * 核心能力：
 *   1. 统一 LLM API 抽象（OpenAI/Anthropic/Google 兼容接口）
 *   2. Agent 运行时（工具调用 + 循环推理）
 *   3. 内置工具集（文件读写/目录操作/Shell 执行）
 *   4. 会话管理（历史记录/状态持久化）
 *   5. CLI 入口（交互式/非交互式模式）
 *
 * 用法：
 *   node pi-mvp.js                          # 交互模式
 *   node pi-mvp.js "帮我写个 hello world"    # 单次提问
 *   node pi-mvp.js --agent coder "重构此函数" # 指定 agent 配置
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

// ============================================================
// 1. 工具函数层 (Utility)
// ============================================================

const Utils = {
  /** 获取 home 目录下的 .pi 配置路径 */
  configPath(...segments) {
    return path.join(os.homedir(), '.pi', ...segments);
  },

  /** 确保目录存在 */
  ensureDir(dir) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    return dir;
  },

  /** 读取 JSON 文件 */
  readJSON(filePath, fallback = null) {
    try {
      const raw = fs.readFileSync(filePath, 'utf-8');
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  },

  /** 写入 JSON 文件（美化输出） */
  writeJSON(filePath, data) {
    this.ensureDir(path.dirname(filePath));
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
  },

  /** 生成唯一 ID (时间戳 + 随机数) */
  uid() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  },

  /** 安全拼接路径，防止目录穿越 */
  safeResolve(base, target) {
    const resolved = path.resolve(base, target);
    if (!resolved.startsWith(path.resolve(base))) {
      throw new Error(`路径越界: ${target}`);
    }
    return resolved;
  }
};

// ============================================================
// 2. LLM Provider 抽象层 (pi-ai 精简版)
// ============================================================

/**
 * BaseProvider — 所有 LLM 提供商的基类
 * 子类只需实现 request() 方法
 */
class BaseProvider {
  constructor(config = {}) {
    this.name = config.name || 'base';
    this.apiKey = config.apiKey || '';
    this.baseUrl = config.baseUrl || '';
    this.model = config.model || 'default';
    this.timeout = config.timeout || 60000; // 60s
  }

  /** 发送聊天请求（子类必须实现） */
  async chat(messages, options = {}) {
    throw new Error('子类必须实现 chat() 方法');
  }

  /** 流式请求（默认回退到非流式） */
  async *stream(messages, options = {}) {
    const response = await this.chat(messages, options);
    yield response;
  }

  /** 列出可用模型 */
  listModels() {
    return [this.model];
  }

  /** 验证配置是否有效 */
  validate() {
    if (!this.apiKey) throw new Error(`${this.name}: 缺少 API Key`);
    return true;
  }
}

/**
 * OpenAI 兼容 Provider
 * 支持：OpenAI / DeepSeek / Ollama / vLLM / 任何 OpenAI 兼容 API
 */
class OpenAICompatible extends BaseProvider {
  constructor(config) {
    super({ name: 'openai-compatible', ...config });
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
    this.model = config.model || 'gpt-4o-mini';
  }

  async chat(messages, options = {}) {
    this.validate();
    const body = {
      model: this.model,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 4096,
      ...(options.tools ? { tools: options.tools } : {})
    };

    const res = await this._fetch('/chat/completions', body);
    const choice = res.choices?.[0];
    if (!choice) throw new Error('API 返回空响应');

    return {
      id: res.id,
      content: choice.message?.content || '',
      toolCalls: choice.message?.tool_calls || null,
      usage: res.usage,
      model: res.model
    };
  }

  async *stream(messages, options = {}) {
    this.validate();
    const body = {
      model: this.model,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 4096
    };

    const res = await this._rawFetch('/chat/completions', body);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // 保留不完整的行

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const data = trimmed.slice(6).trim();
        if (data === '[DONE]') return;

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta;
          if (delta?.content) yield { type: 'content', text: delta.content };
        } catch { /* 忽略解析错误 */ }
      }
    }
  }

  async _fetch(endpoint, body) {
    const res = await this._rawFetch(endpoint, body);
    return await res.json();
  }

  _rawFetch(endpoint, body) {
    return fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout)
    });
  }

  listModels() {
    return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'deepseek-chat', 'llama3'];
  }
}

/**
 * Anthropic Claude Provider
 */
class AnthropicProvider extends BaseProvider {
  constructor(config) {
    super({ name: 'anthropic', ...config });
    this.baseUrl = config.baseUrl || 'https://api.anthropic.com';
    this.model = config.model || 'claude-sonnet-4-20250514';
  }

  async chat(messages, options = {}) {
    this.validate();

    // 将消息转换为 Anthropic 格式（system/user/assistant 分离）
    const systemMsgs = messages.filter(m => m.role === 'system');
    const system = systemMsgs.map(m => m.content).join('\n');
    const otherMsgs = messages.filter(m => m.role !== 'system').map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content
    }));

    // Anthropic 要求 user/assistant 交替，以 user 结尾
    if (otherMsgs.length === 0 || otherMsgs[otherMsgs.length - 1].role !== 'user') {
      otherMsgs.push({ role: 'user', content: '(继续)' });
    }

    const body = {
      model: this.model,
      max_tokens: options.maxTokens ?? 4096,
      system,
      messages: otherMsgs,
      ...(options.tools ? { tools: options.tools } : {})
    };

    const res = await fetch(`${this.baseUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout)
    });

    const data = await res.json();
    if (data.error) throw new Error(`Anthropic API 错误: ${data.error.message}`);

    const block = data.content?.[0];
    return {
      id: data.id,
      content: block?.type === 'text' ? block.text : '',
      toolCalls: data.content?.filter(b => b.type === 'tool_use')?.map(b => ({
        id: b.id,
        name: b.name,
        arguments: b.input
      })) || null,
      usage: data.usage,
      model: data.model
    };
  }

  listModels() {
    return ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-haiku-3-5-20241022'];
  }
}

/**
 * Google Gemini Provider
 */
class GoogleProvider extends BaseProvider {
  constructor(config) {
    super({ name: 'google', ...config });
    this.baseUrl = config.baseUrl || 'https://generativelanguage.googleapis.com';
    this.model = config.model || 'gemini-2.0-flash';
  }

  async chat(messages, options = {}) {
    this.validate();

    // 将消息转换为 Gemini 格式
    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

    const systemInstruction = messages
      .filter(m => m.role === 'system')
      .map(m => m.content)
      .join('\n');

    const body = {
      contents,
      ...(systemInstruction ? { systemInstruction: { parts: [{ text: systemInstruction }] } } {}),
      generationConfig: {
        temperature: options.temperature ?? 0.7,
        maxOutputTokens: options.maxTokens ?? 4096
      }
    };

    const url = `${this.baseUrl}/v1beta/models/${this.model}:generateContent?key=${this.apiKey}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout)
    });

    const data = await res.json();
    if (data.error) throw new Error(`Google API 错误: ${data.error.message}`);

    const candidate = data.candidates?.[0];
    const text = candidate?.content?.parts?.[0]?.text || '';

    return {
      id: data.metadata?.responseId || Utils.uid(),
      content: text,
      toolCalls: null,
      usage: data.usageMetadata,
      model: this.model
    };
  }

  listModels() {
    return ['gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-flash', 'gemini-1.5-pro'];
  }
}

// ============================================================
// 3. 工具系统 (Tool System)
// ============================================================

/**
 * Tool — 工具基类
 * 每个 Tool 有名称、描述、参数 schema 和执行函数
 */
class Tool {
  constructor(def) {
    this.name = def.name;
    this.description = def.description;
    this.parameters = def.parameters || {};
    this.execute = def.execute; // async function(params) => result
  }

  /** 转换为 OpenAI function calling 格式 */
  toOpenAIFunc() {
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters
      }
    };
  }

  /** 转换为 Anthropic tool 格式 */
  toAnthropicTool() {
    return {
      name: this.name,
      description: this.description,
      input_schema: this.parameters
    };
  }
}

/** 内置工具集 */
const BuiltinTools = {
  /**
   * 读取文件内容
   */
  read_file: new Tool({
    name: 'read_file',
    description: '读取文件内容，返回完整文本。支持相对路径和绝对路径。',
    parameters: {
      type: 'object',
      properties: {
        filePath: { type: 'string', description: '要读取的文件路径' },
        encoding: { type: 'string', default: 'utf-8', description: '文件编码' }
      },
      required: ['filePath']
    },
    async execute({ filePath, encoding = 'utf-8' }, context) {
      const resolved = path.isAbsolute(filePath)
        ? filePath
        : path.join(context.workDir || process.cwd(), filePath);

      if (!fs.existsSync(resolved)) {
        return { error: `文件不存在: ${resolved}` };
      }

      const stat = fs.statSync(resolved);
      if (stat.size > 1024 * 500) { // 限制 500KB
        return { error: `文件过大 (${(stat.size / 1024).toFixed(0)}KB)，请分段读取或使用其他方式` };
      }

      const content = fs.readFileSync(resolved, encoding);
      return {
        success: true,
        path: resolved,
        content,
        size: stat.size,
        lines: content.split('\n').length
      };
    }
  }),

  /**
   * 写入文件
   */
  write_file: new Tool({
    name: 'write_file',
    description: '将内容写入文件。会自动创建父目录。覆盖已有文件。',
    parameters: {
      type: 'object',
      properties: {
        filePath: { type: 'string', description: '目标文件路径' },
        content: { type: 'string', description: '要写入的内容' },
        encoding: { type: 'string', default: 'utf-8', description: '文件编码' }
      },
      required: ['filePath', 'content']
    },
    async execute({ filePath, content, encoding = 'utf-8' }, context) {
      const resolved = path.isAbsolute(filePath)
        ? filePath
        : path.join(context.workDir || process.cwd(), filePath);

      Utils.ensureDir(path.dirname(resolved));
      fs.writeFileSync(resolved, content, encoding);

      return {
        success: true,
        path: resolved,
        size: Buffer.byteLength(content, encoding),
        message: `已写入 ${resolved}`
      };
    }
  }),

  /**
   * 列出目录内容
   */
  list_dir: new Tool({
    name: 'list_dir',
    description: '列出目录中的文件和子目录。',
    parameters: {
      type: 'object',
      properties: {
        dirPath: { type: 'string', description: '目录路径（默认当前工作目录）' },
        recursive: { type: 'boolean', default: false, description: '是否递归列出' },
        pattern: { type: 'string', default: '*', description: '匹配模式（如 *.ts）' }
      },
      required: []
    },
    async execute({ dirPath, recursive = false, pattern = '*' }, context) {
      const baseDir = dirPath
        ? (path.isAbsolute(dirPath) ? dirPath : path.join(context.workDir || process.cwd(), dirPath))
        : (context.workDir || process.cwd());

      if (!fs.existsSync(baseDir)) {
        return { error: `目录不存在: ${baseDir}` };
      }

      const items = [];
      const scan = (dir, depth = 0) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relPath = path.relative(baseDir, fullPath);

          // 简单的模式匹配
          if (pattern !== '*') {
            const regex = new RegExp('^' + pattern.replace(/\*/g, '.*').replace(/\?/g, '.') + '$');
            if (!regex.test(entry.name)) continue;
          }

          items.push({
            name: entry.name,
            path: relPath || entry.name,
            type: entry.isDirectory() ? 'directory' : 'file',
            depth
          });

          if (entry.isDirectory() && recursive && depth < 10) {
            scan(fullPath, depth + 1);
          }
        }
      };

      scan(baseDir);
      return { success: true, path: baseDir, count: items.length, items };
    }
  }),

  /**
   * 执行 Shell 命令
   */
  exec_command: new Tool({
    name: 'exec_command',
    description: '执行 Shell 命令并返回输出。注意安全风险，谨慎使用。',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: '要执行的命令' },
        cwd: { type: 'string', description: '工作目录（可选）' },
        timeout: { type: 'number', default: 30000, description: '超时时间(ms)' }
      },
      required: ['command']
    },
    async execute({ command, cwd, timeout = 30000 }, context) {
      const { execSync } = require('child_process');
      try {
        const workDir = cwd || context.workDir || process.cwd();
        const stdout = execSync(command, {
          cwd: workDir,
          encoding: 'utf-8',
          timeout,
          stdio: ['pipe', 'pipe', 'pipe'],
          maxBuffer: 1024 * 500
        });
        return { success: true, stdout, exitCode: 0 };
      } catch (err) {
        return {
          success: false,
          stdout: err.stdout || '',
          stderr: err.stderr || err.message,
          exitCode: err.status || 1
        };
      }
    }
  }),

  /**
   * 搜索文件内容
   */
  search_files: new Tool({
    name: 'search_files',
    description: '在指定目录中搜索包含特定文本或正则表达式的文件。',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '搜索关键词或正则表达式' },
        dirPath: { type: 'string', description: '搜索目录（默认当前目录）' },
        filePattern: { type: 'string', default: '*', description: '文件名过滤模式' },
        maxResults: { type: 'number', default: 20, description: '最大结果数' }
      },
      required: ['query']
    },
    async execute({ query, dirPath, filePattern = '*', maxResults = 20 }, context) {
      const baseDir = dirPath
        ? (path.isAbsolute(dirPath) ? dirPath : path.join(context.workDir || process.cwd(), dirPath))
        : (context.workDir || process.cwd());

      const results = [];
      const regex = new RegExp(query, 'i');
      const walk = (dir) => {
        if (!fs.existsSync(dir)) return;
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            walk(fullPath);
            continue;
          }
          // 文件名过滤
          if (filePattern !== '*') {
            const fpRegex = new RegExp('^' + filePattern.replace(/\*/g, '.*') + '$');
            if (!fpRegex.test(entry.name)) continue;
          }
          try {
            const content = fs.readFileSync(fullPath, 'utf-8');
            const lines = content.split('\n');
            for (let i = 0; i < lines.length && results.length < maxResults; i++) {
              if (regex.test(lines[i])) {
                results.push({
                  file: path.relative(baseDir, fullPath),
                  line: i + 1,
                  match: lines[i].trim().slice(0, 200)
                });
              }
            }
          } catch { /* 二进制文件等跳过 */ }
        }
      };

      walk(baseDir);
      return { success: true, query, total: results.length, results };
    }
  }),

  /**
   * 获取文件信息
   */
  file_info: new Tool({
    name: 'file_info',
    description: '获取文件的元信息（大小、修改时间、类型等）。',
    parameters: {
      type: 'object',
      properties: {
        filePath: { type: 'string', description: '文件或目录路径' }
      },
      required: ['filePath']
    },
    async execute({ filePath }, context) {
      const resolved = path.isAbsolute(filePath)
        ? filePath
        : path.join(context.workDir || process.cwd(), filePath);

      if (!fs.existsSync(resolved)) {
        return { error: `不存在: ${resolved}` };
      }

      const stat = fs.statSync(resolved);
      return {
        success: true,
        path: resolved,
        isDirectory: stat.isDirectory(),
        isFile: stat.isFile(),
        size: stat.size,
        created: stat.birthtime.toISOString(),
        modified: stat.mtime.toISOString(),
        permissions: stat.mode.toString(8).slice(-3)
      };
    }
  }),

  /**
   * 思考/规划（让 Agent 可以内部推理）
   */
  think: new Tool({
    name: 'think',
    description: '用于内部思考和规划。不会产生副作用，可以多次调用来组织思路。',
    parameters: {
      type: 'object',
      properties: {
        thought: { type: 'string', description: '思考内容' }
      },
      required: ['thought']
    },
    async execute({ thought }) {
      return { success: true, thought, timestamp: new Date().toISOString() };
    }
  })
};

/** 获取所有内置工具 */
function getAllTools() {
  return Object.values(BuiltinTools);
}

/** 按名称查找工具 */
function getToolByName(name) {
  return BuiltinTools[name] || null;
}

// ============================================================
// 4. Agent 运行时 (Agent Runtime)
// ============================================================

/**
 * Session — 会话管理
 */
class Session {
  constructor(options = {}) {
    this.id = options.id || Utils.uid();
    this.createdAt = new Date().toISOString();
    this.messages = [];           // 对话历史
    this.state = {};              // 自定义状态
    this.metadata = options.metadata || {};
    this.workDir = options.workDir || process.cwd();
    this.toolResults = [];        // 工具调用记录
    this.maxRounds = options.maxRounds || 30; // 最大轮次
    this.roundCount = 0;
  }

  /** 添加消息 */
  addMessage(role, content) {
    this.messages.push({
      role,
      content: typeof content === 'string' ? content : JSON.stringify(content),
      timestamp: new Date().toISOString()
    });
  }

  /** 获取格式化后的对话历史（供 LLM 使用） */
  getHistory() {
    return this.messages.map(m => ({ role: m.role, content: m.content }));
  }

  /** 更新状态 */
  setState(key, value) {
    this.state[key] = value;
  }

  /** 序列化 */
  toJSON() {
    return {
      id: this.id,
      createdAt: this.createdAt,
      messageCount: this.messages.length,
      state: this.state,
      metadata: this.metadata,
      workDir: this.workDir,
      toolResultCount: this.toolResults.length
    };
  }
}

/**
 * Agent — 核心 Agent 引擎
 * 实现 ReAct 模式的工具调用循环
 */
class Agent {
  constructor(config = {}) {
    this.provider = config.provider;       // LLM Provider 实例
    this.tools = config.tools || getAllTools();
    this.systemPrompt = config.systemPrompt || DEFAULT_SYSTEM_PROMPT;
    this.session = config.session || new Session({ workDir: config.workDir });
    this.verbose = config.verbose ?? true;
    this.maxIterations = config.maxIterations || 30;
  }

  /** 设置系统提示词 */
  setSystemPrompt(prompt) {
    this.systemPrompt = prompt;
    return this;
  }

  /** 运行一次完整的 Agent 循环 */
  async run(userInput) {
    // 1. 添加用户消息
    this.session.addMessage('user', userInput);

    // 2. 构建带 system prompt 的消息列表
    const messages = [
      { role: 'system', content: this.systemPrompt },
      ...this.session.getHistory()
    ];

    // 3. 构建工具定义（根据 provider 类型）
    const tools = this._buildToolDefinitions();

    // 4. 开始 Agent 循环
    for (let i = 0; i < this.maxIterations; i++) {
      this.session.roundCount++;

      if (this.verbose) {
        console.log(`\n${'─'.repeat(60)}`);
        console.log(`🔄 Agent 第 ${this.session.roundCount} 轮`);
        console.log(`${'─'.repeat(60)}`);
      }

      try {
        // 调用 LLM
        const response = await this.provider.chat(messages, { tools });

        // 处理响应
        if (response.toolCalls && response.toolCalls.length > 0) {
          // 有工具调用 → 执行工具并反馈结果
          this.session.addMessage('assistant', response.content || '(正在调用工具...)');

          for (const tc of response.toolCalls) {
            const toolName = tc.name;
            const args = typeof tc.arguments === 'string'
              ? JSON.parse(tc.arguments)
              : tc.arguments;

            if (this.verbose) {
              console.log(`  🔧 工具调用: ${toolName}(${JSON.stringify(args).slice(0, 100)})`);
            }

            const tool = getToolByName(toolName);
            if (!tool) {
              const errorMsg = `未知工具: ${toolName}`;
              if (this.verbose) console.log(`  ❌ ${errorMsg}`);
              this._appendToolResult(toolName, args, { error: errorMsg }, messages);
              continue;
            }

            try {
              const result = await tool.execute(args, {
                workDir: this.session.workDir,
                session: this.session
              });

              if (this.verbose) {
                const preview = JSON.stringify(result).slice(0, 200);
                console.log(`  ✅ 结果: ${preview}${result.length > 200 ? '...' : ''}`);
              }

              this.session.toolResults.push({ toolName, args, result, round: this.session.roundCount });
              this._appendToolResult(toolName, args, result, messages);
            } catch (err) {
              const errorResult = { error: err.message, stack: err.stack };
              if (this.verbose) console.log(`  ❌ 错误: ${err.message}`);
              this._appendToolResult(toolName, args, errorResult, messages);
            }
          }
        } else {
          // 无工具调用 → 最终回复
          this.session.addMessage('assistant', response.content);
          return {
            success: true,
            content: response.content,
            rounds: this.session.roundCount,
            toolCalls: this.session.toolResults.length,
            session: this.session.toJSON()
          };
        }
      } catch (err) {
        if (this.verbose) console.log(`  💥 异常: ${err.message}`);
        // 如果是网络错误等，尝试返回已有上下文
        if (i >= 2) {
          return {
            success: false,
            error: err.message,
            partial: true,
            rounds: this.session.roundCount,
            session: this.session.toJSON()
          };
        }
        throw err;
      }
    }

    return {
      success: false,
      error: '达到最大迭代次数',
      rounds: this.session.roundCount,
      session: this.session.toJSON()
    };
  }

  /** 将工具结果追加到消息列表 */
  _appendToolResult(toolName, args, result, messages) {
    // OpenAI 格式的 tool response
    messages.push({
      role: 'tool',
      tool_call_id: `${toolName}-${Date.now()}`,
      name: toolName,
      content: JSON.stringify(result)
    });
  }

  /** 根据 provider 类型构建工具定义 */
  _buildToolDefinitions() {
    if (this.provider instanceof AnthropicProvider) {
      return this.tools.map(t => t.toAnthropicTool());
    }
    return this.tools.map(t => t.toOpenAIFunc());
  }
}

// ============================================================
// 5. 默认系统提示词
// ============================================================

const DEFAULT_SYSTEM_PROMPT = `你是一个高效的 AI 编程助手（Pi-Agent），具备以下能力：

## 可用工具
- read_file: 读取文件内容
- write_file: 写入文件
- list_dir: 列出目录
- exec_command: 执行 Shell 命令
- search_files: 搜索文件内容
- file_info: 获取文件信息
- think: 内部思考与规划

## 工作原则
1. **理解优先** — 先用 think 工具分析需求，再动手
2. **分步执行** — 复杂任务拆分为小步骤，逐步完成
3. **先读后写** — 修改文件前先用 read_file 了解现有代码
4. **确认变更** — 重要操作前说明计划
5. **简洁高效** — 直接给出可运行的代码，避免冗余解释

## 输出规范
- 代码块标注语言类型
- 修改文件时说明改动原因
- 遇到错误时提供修复建议`;

// ============================================================
// 6. 配置管理 (Config)
// ============================================================

const ConfigManager = {
  /** 默认配置 */
  defaults: {
    provider: 'openai',
    model: 'gpt-4o-mini',
    apiKey: '',
    baseUrl: '',
    workDir: process.cwd(),
    maxIterations: 30,
    verbose: true
  },

  /** 加载配置（按优先级：环境变量 > 配置文件 > 默认值） */
  load(customOverrides = {}) {
    // 从 ~/.pi/config.json 读取
    const configFile = Utils.configPath('config.json');
    const fileConfig = Utils.readJSON(configFile, {});

    // 从环境变量读取
    const envConfig = {};
    if (process.env.PI_API_KEY) envConfig.apiKey = process.env.PI_API_KEY;
    if (process.env.PI_PROVIDER) envConfig.provider = process.env.PI_PROVIDER;
    if (process.env.PI_MODEL) envConfig.model = process.env.PI_MODEL;
    if (process.env.PI_BASE_URL) envConfig.baseUrl = process.env.PI_BASE_URL;
    if (process.env.OPENAI_API_KEY && !envConfig.apiKey) envConfig.apiKey = process.env.OPENAI_API_KEY;
    if (process.env.ANTHROPIC_API_KEY && !envConfig.apiKey) envConfig.apiKey = process.env.ANTHROPIC_API_KEY;
    if (process.env.GOOGLE_API_KEY && !envConfig.apiKey) envConfig.apiKey = process.env.GOOGLE_API_KEY;

    // 合并配置
    return { ...this.defaults, ...fileConfig, ...envConfig, ...customOverrides };
  },

  /** 保存配置 */
  save(config) {
    const filePath = Utils.configPath('config.json');
    Utils.writeJSON(filePath, config);
    return filePath;
  },

  /** 创建 Provider 实例 */
  createProvider(config) {
    switch (config.provider) {
      case 'anthropic':
        return new AnthropicProvider(config);
      case 'google':
        return new GoogleProvider(config);
      case 'openai':
      default:
        return new OpenAICompatible(config);
    }
  }
};

// ============================================================
// 7. CLI 入口
// ============================================================

/**
 * 解析命令行参数
 */
function parseArgs(argv) {
  const args = {
    prompt: null,
    provider: null,
    model: null,
    apiKey: null,
    baseUrl: null,
    workDir: null,
    nonInteractive: false,
    help: false,
    version: false,
    saveConfig: false,
    listModels: false
  };

  let i = 2; // 跳过 node 和脚本路径
  while (i < argv.length) {
    const arg = argv[i];

    if (arg === '-h' || arg === '--help') {
      args.help = true;
    } else if (arg === '-v' || arg === '--version') {
      args.version = true;
    } else if (arg === '-p' || arg === '--provider') {
      args.provider = argv[++i];
    } else if (arg === '-m' || arg === '--model') {
      args.model = argv[++i];
    } else if (arg === '--api-key') {
      args.apiKey = argv[++i];
    } else if (arg === '--base-url') {
      args.baseUrl = argv[++i];
    } else if (arg === '-d' || arg === '--work-dir') {
      args.workDir = argv[++i];
    } else if (arg === '-n' || arg === '--non-interactive') {
      args.nonInteractive = true;
    } else if (arg === '--save-config') {
      args.saveConfig = true;
    } else if (arg === '--list-models') {
      args.listModels = true;
    } else if (!arg.startsWith('-')) {
      args.prompt = arg;
    }

    i++;
  }

  return args;
}

/** 显示帮助信息 */
function showHelp() {
  console.log(`
╔══════════════════════════════════════════════╗
║         Pi-MVP  AI Agent Toolkit v0.1        ║
║     单文件版 · 仅依赖 fs/path/os             ║
╚══════════════════════════════════════════════╝

用法:
  node pi-mvp.js [选项] [prompt]

模式:
  无参数          交互模式（REPL）
  "你的问题"      单次问答模式

选项:
  -p, --provider <name>    LLM 提供商 (openai|anthropic|google)
                           默认: openai
  -m, --model <name>       模型名称
                           默认: gpt-4o-mini
  --api-key <key>          API 密钥（也可用 PI_API_KEY 环境变量）
  --base-url <url>         API 基础 URL（用于兼容 API）
  -d, --work-dir <path>    工作目录
  -n, --non-interactive    非交互模式（自动退出）
  --list-models            列出可用模型
  --save-config            保存当前配置到 ~/.pi/config.json
  -h, --help               显示帮助
  -v, --version            显示版本

环境变量:
  PI_API_KEY      API 密钥
  PI_PROVIDER     提供商 (openai/anthropic/google)
  PI_MODEL        模型名称
  PI_BASE_URL     API 基础 URL
  OPENAI_API_KEY  OpenAI API Key（备用）
  ANTHROPIC_API_KEY  Anthropic API Key（备用）

示例:
  node pi-mvp.js "用 TypeScript 写一个快排"
  node pi-mvp.js -p anthropic -m claude-haiku "解释这段代码"
  node pi-mvp.js --base-url http://localhost:11434/v1 -p openai -m llama3 "你好"
  node pi-mvp.js --save-config -p deepseek -m deepseek-chat --api-key sk-xxx

配置文件: ~/.pi/config.json
`);
}

/** 显示版本 */
function showVersion() {
  console.log('Pi-MVP v0.1.0 — Single-file AI Agent Toolkit');
  console.log('仅依赖: fs, path, os');
}

/** 列出模型 */
function listModels(provider) {
  const p = ConfigManager.createProvider({ ...ConfigManager.defaults, ...{ provider: provider || 'openai' } });
  console.log(`\n📋 ${p.name} 可用模型:\n`);
  p.listModels().forEach((m, i) => console.log(`  ${i + 1}. ${m}`));
}

/** 交互式 REPL */
async function startRepl(agent) {
  const readline = require('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  console.log('\n🚀 Pi-MVP 交互模式启动！');
  console.log('   输入问题开始对话，输入 /exit 退出，/clear 清空上下文\n');

  const prompt = () => {
    return new Promise(resolve => rl.question('❯ ', resolve));
  };

  while (true) {
    try {
      const input = await prompt();

      if (!input.trim()) continue;

      // 内置命令
      if (input === '/exit' || input === '/quit') {
        console.log('👋 再见！');
        break;
      }
      if (input === '/clear') {
        agent.session = new Session({ workDir: agent.session.workDir });
        console.log('✅ 上下文已清空\n');
        continue;
      }
      if (input === '/status') {
        console.log(JSON.stringify(agent.session.toJSON(), null, 2));
        continue;
      }
      if (input.startsWith('/help')) {
        showHelp();
        continue;
      }

      // 运行 Agent
      const result = await agent.run(input);
      console.log(`\n🤖 ${result.content || result.error}\n`);

      if (result.toolCalls > 0) {
        console.log(`   (共调用 ${result.toolCalls} 个工具, ${result.rounds} 轮)\n`);
      }
    } catch (err) {
      console.error(`\n❌ 错误: ${err.message}\n`);
    }
  }

  rl.close();
}

// ============================================================
// 8. 主入口
// ============================================================

async function main() {
  const args = parseArgs(process.argv);

  // 帮助 & 版本
  if (args.help) { showHelp(); process.exit(0); }
  if (args.version) { showVersion(); process.exit(0); }
  if (args.listModels) { listModels(args.provider); process.exit(0); }

  // 加载配置
  const config = ConfigManager.load({
    ...(args.provider && { provider: args.provider }),
    ...(args.model && { model: args.model }),
    ...(args.apiKey && { apiKey: args.apiKey }),
    ...(args.baseUrl && { baseUrl: args.baseUrl }),
    ...(args.workDir && { workDir: args.workDir })
  });

  // 保存配置
  if (args.saveConfig) {
    const saved = ConfigManager.save(config);
    console.log(`✅ 配置已保存: ${saved}`);
    if (!args.prompt) process.exit(0);
  }

  // 验证 API Key
  if (!config.apiKey) {
    console.error('❌ 缺少 API Key！请通过以下方式设置：');
    console.error('   1. 环境变量: export PI_API_KEY=sk-xxx');
    console.error('   2. 参数: node pi-mvp.js --api-key sk-xxx "你好"');
    console.error('   3. 配置文件: ~/.pi/config.json');
    console.error('\n   或使用本地兼容 API:');
    console.error('   node pi-mvp.js --base-url http://localhost:11434/v1 -m llama3 "你好"');
    process.exit(1);
  }

  // 创建 Provider 和 Agent
  const provider = ConfigManager.createProvider(config);
  const agent = new Agent({
    provider,
    workDir: config.workDir,
    maxIterations: config.maxIterations,
    verbose: config.verbose
  });

  // 启动信息
  if (config.verbose) {
    console.log(`\n🔧 Pi-MVP 启动`);
    console.log(`   Provider: ${provider.name}`);
    console.log(`   Model: ${provider.model}`);
    console.log(`   WorkDir: ${config.workDir}`);
  }

  // 运行模式
  if (args.prompt) {
    // 单次模式
    try {
      const result = await agent.run(args.prompt);
      console.log(result.content || result.error || '(无回复)');
      if (config.verbose && result.toolCalls > 0) {
        console.log(`\n📊 工具调用: ${result.toolCalls} 次 | 轮次: ${result.rounds}`);
      }
    } catch (err) {
      console.error(`❌ ${err.message}`);
      process.exit(1);
    }
  } else {
    // 交互模式
    await startRepl(agent);
  }
}

// 导出模块（支持 require 复用）
module.exports = {
  // Providers
  BaseProvider,
  OpenAICompatible,
  AnthropicProvider,
  GoogleProvider,

  // Tools
  Tool,
  BuiltinTools,
  getAllTools,
  getToolByName,

  // Agent
  Session,
  Agent,

  // Config
  ConfigManager,
  DEFAULT_SYSTEM_PROMPT,

  // Utility
  Utils,

  // CLI
  parseArgs,
  main
};

// 直接运行时执行 main
if (require.main === module) {
  main().catch(err => {
    console.error('💥 致命错误:', err.message);
    process.exit(1);
  });
}
