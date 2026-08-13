/**
 * Pi-MVP v0.4 — 单文件 AI Agent Toolkit + MCP + Skills 系统
 *
 * 核心能力：
 *   1. 多 LLM 路由配置（llm/llm_router/llm_planner/llm_profile）
 *   2. 统一 LLM API 抽象（OpenAI/Anthropic/Google 兼容接口）
 *   3. Agent 运行时（工具调用 + 循环推理 + 意图路由）
 *   4. 内置工具集（文件读写/目录操作）
 *   5. MCP (Model Context Protocol) — Streamable HTTP 传输
 *   6. Skills 系统 — 按需加载，意图驱动，节约上下文
 *   7. 会话管理（历史记录/状态持久化）
 *   8. CLI 入口（交互式/非交互式模式）
 *
 * 用法：
 *   node pi-mvp.js                          # 交互模式
 *   node pi-mvp.js "帮我写个 hello world"    # 单次提问
 *   node pi-mvp.js --skill-create my-skill   # 创建新技能
 *
 * 依赖模块: fs, path, os (全部 Node.js 内置)
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

// ============================================================
// 0. 工作空间 & 全局路径
// ============================================================

const WORKSPACE_DIR = path.resolve(__dirname, 'workspace');
const SKILLS_DIR = path.join(WORKSPACE_DIR, 'skills');
const WORKSPACE_CONFIG_PATH = path.join(WORKSPACE_DIR, 'config.json');

const Utils = {
  configPath(...segments) {
    return path.join(os.homedir(), '.pi', ...segments);
  },

  workspacePath(...segments) {
    return path.join(WORKSPACE_DIR, ...segments);
  },

  ensureDir(dir) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    return dir;
  },

  readJSON(filePath, fallback = null) {
    try {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } catch {
      return fallback;
    }
  },

  writeJSON(filePath, data) {
    this.ensureDir(path.dirname(filePath));
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
  },

  uid() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  },

  safeResolve(base, target) {
    const resolved = path.resolve(base, target);
    if (!resolved.startsWith(path.resolve(base))) {
      throw new Error(`路径越界: ${target}`);
    }
    return resolved;
  }
};

// ============================================================
// 1. LLM Provider 抽象层 (pi-ai 精简版)
// ============================================================

class BaseProvider {
  constructor(config = {}) {
    this.name = config.name || 'base';
    this.apiKey = config.apiKey || '';
    this.baseUrl = config.baseUrl || '';
    this.model = config.model || 'default';
    this.timeout = config.timeout || 60000;
    this.streaming = config.streaming || false;
  }

  async chat(messages, options = {}) {
    throw new Error('子类必须实现 chat() 方法');
  }

  async *stream(messages, options = {}) {
    const response = await this.chat(messages, options);
    yield response;
  }

  listModels() { return [this.model]; }

  validate() {
    if (!this.apiKey) throw new Error(`${this.name}: 缺少 API Key`);
    return true;
  }
}

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

  async _fetch(endpoint, body) {
    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout)
    });
    return await res.json();
  }

  listModels() {
    return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'deepseek-chat', 'llama3'];
  }
}

class AnthropicProvider extends BaseProvider {
  constructor(config) {
    super({ name: 'anthropic', ...config });
    this.baseUrl = config.baseUrl || 'https://api.anthropic.com';
    this.model = config.model || 'claude-sonnet-4-20250514';
  }

  async chat(messages, options = {}) {
    this.validate();

    const systemMsgs = messages.filter(m => m.role === 'system');
    const system = systemMsgs.map(m => m.content).join('\n');
    const otherMsgs = messages.filter(m => m.role !== 'system').map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content
    }));

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
        id: b.id, name: b.name, arguments: b.input
      })) || null,
      usage: data.usage,
      model: data.model
    };
  }

  listModels() {
    return ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-haiku-3-5-20241022'];
  }
}

class GoogleProvider extends BaseProvider {
  constructor(config) {
    super({ name: 'google', ...config });
    this.baseUrl = config.baseUrl || 'https://generativelanguage.googleapis.com';
    this.model = config.model || 'gemini-2.0-flash';
  }

  async chat(messages, options = {}) {
    this.validate();

    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

    const systemInstruction = messages
      .filter(m => m.role === 'system')
      .map(m => m.content).join('\n');

    const body = {
      contents,
      ...(systemInstruction ? { systemInstruction: { parts: [{ text: systemInstruction }] } } : {}),
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
    return {
      id: data.metadata?.responseId || Utils.uid(),
      content: candidate?.content?.parts?.[0]?.text || '',
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
// 2. 多 LLM 路由器 (LLM Router)
// ============================================================

/**
 * LLMRouter — 多 Provider 路由管理器
 *
 * 支持为不同场景指定不同的 LLM 配置：
 *   - llm:       默认对话模型
 *   - llm_router: 意图路由/分类模型（通常用轻量快速模型）
 *   - llm_planner: 任务规划模型（需要强推理能力）
 *   - llm_profile: 用户画像/个性化模型
 *
 * 配置来源优先级：workspace/config.json > ~/.pi/config.json > 默认值
 */
class LLMRouter {
  constructor(config = {}) {
    this.config = config;
    this.providers = {};
    this._initProviders();
  }

  _initProviders() {
    // 从配置创建各角色 Provider
    const roles = ['llm', 'llm_router', 'llm_planner', 'llm_profile'];
    for (const role of roles) {
      const cfg = this.config[role];
      if (cfg && cfg.api_key) {
        this.providers[role] = this._createProvider(cfg, role);
      }
    }

    // 确保 llm 至少存在
    if (!this.providers.llm) {
      const defaultCfg = this.config.llm || {
        base_url: 'https://api.openai.com/v1',
        api_key: '',
        model: 'gpt-4o-mini'
      };
      this.providers.llm = this._createProvider(defaultCfg, 'llm');
    }
  }

  _createProvider(cfg, role) {
    const baseUrl = cfg.base_url || cfg.baseUrl || '';
    // 根据 base_url 自动检测 provider 类型
    let type = 'openai';
    if (baseUrl.includes('anthropic') || baseUrl.includes('claude')) type = 'anthropic';
    else if (baseUrl.includes('google') || baseUrl.includes('generativelanguage')) type = 'google';

    const providerConfig = {
      name: `${role}-${type}`,
      apiKey: cfg.api_key || cfg.apiKey || '',
      baseUrl: baseUrl,
      model: cfg.model || 'default',
      streaming: cfg.streaming || false,
      timeout: cfg.timeout || 60000
    };

    switch (type) {
      case 'anthropic': return new AnthropicProvider(providerConfig);
      case 'google': return new GoogleProvider(providerConfig);
      default: return new OpenAICompatible(providerConfig);
    }
  }

  /** 获取指定角色的 Provider，fallback 到默认 llm */
  get(role = 'llm') {
    return this.providers[role] || this.providers.llm;
  }

  /** 获取默认对话 Provider */
  get chat() { return this.get('llm'); }

  /** 获取意图路由 Provider */
  get router() { return this.get('llm_router') || this.get('llm'); }

  /** 获取任务规划 Provider */
  get planner() { return this.get('llm_planner') || this.get('llm'); }

  /** 获取画像 Provider */
  get profile() { return this.get('llm_profile') || this.get('llm'); }

  /** 所有已配置的角色列表 */
  get roles() {
    return Object.keys(this.providers);
  }

  /** 获取所有 Provider 的状态摘要 */
  getStatus() {
    return Object.fromEntries(
      Object.entries(this.providers).map(([role, provider]) => [
        role,
        { name: provider.name, model: provider.model, baseUrl: provider.baseUrl }
      ])
    );
  }
}

// ============================================================
// 3. MCP (Model Context Protocol) — Streamable HTTP 实现
// ============================================================

const MCPProtocol = {
  JSONRPC_VERSION: '2.0',
  PROTOCOL_VERSION: '2024-11-05',

  Methods: {
    INITIALIZE: 'initialize',
    INITIALIZED_NOTIFICATION: 'notifications/initialized',
    LIST_TOOLS: 'tools/list',
    CALL_TOOL: 'tools/call',
    LIST_RESOURCES: 'resources/list',
    READ_RESOURCE: 'resources/read',
    LIST_PROMPTS: 'prompts/list',
    GET_PROMPT: 'prompts/get',
    PING: 'ping'
  }
};

class MCPHttpTransport {
  constructor(baseUrl, headers = {}, options = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
      ...headers
    };
    this.timeout = options.timeout || 30000;
    this.requestId = 0;
    this.sessionId = null;
    this._closed = false;
  }

  async request(method, params = {}, timeoutMs) {
    if (this._closed) throw new Error('MCP Transport 已关闭');

    const id = ++this.requestId;
    const body = {
      jsonrpc: MCPProtocol.JSONRPC_VERSION,
      id,
      method,
      params
    };

    const reqHeaders = { ...this.headers };
    if (this.sessionId) reqHeaders['Mcp-Session-Id'] = this.sessionId;

    const ms = timeoutMs || this.timeout;

    try {
      const res = await fetch(this.baseUrl, {
        method: 'POST',
        headers: reqHeaders,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(ms)
      });

      const sid = res.headers.get('mcp-session-id');
      if (sid) this.sessionId = sid;

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
      }

      const data = await res.json();
      if (data.error) throw new Error(`MCP Error [${data.error.code}]: ${data.error.message}`);
      return data.result;
    } catch (err) {
      if (err.name === 'TimeoutError' || err.message.includes('abort')) {
        throw new Error(`MCP 请求超时: ${method} (${ms}ms)`);
      }
      throw err;
    }
  }

  async notify(method, params = {}) {
    if (this._closed) return;
    const body = { jsonrpc: MCPProtocol.JSONRPC_VERSION, method, params };
    const reqHeaders = { ...this.headers };
    if (this.sessionId) reqHeaders['Mcp-Session-Id'] = this.sessionId;
    try {
      await fetch(this.baseUrl, {
        method: 'POST',
        headers: reqHeaders,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(5000)
      }).catch(() => {});
    } catch { /* ignore */ }
  }

  close() { this._closed = true; }
  get closed() { return this._closed; }
  get endpoint() { return this.baseUrl; }
}

class MCPServer {
  constructor(name, config, options = {}) {
    this.name = name;
    this.config = config;
    this.verbose = options.verbose ?? true;
    this.transport = null;
    this.serverInfo = null;
    this.capabilities = {};
    this.tools = [];
    this.resources = [];
    this.prompts = [];
    this._initialized = false;
    this._connectTime = null;
    this._callCount = 0;
  }

  get isReady() { return this._initialized && this.transport && !this.transport.closed; }
  get uptime() { return this._connectTime ? Date.now() - this._connectTime : 0; }

  async connect() {
    if (this.isReady) { if (this.verbose) console.log(`  🔄 [${this.name}] 已连接，跳过`); return this; }

    try {
      const url = this.config.url;
      if (!url) throw new Error(`[${this.name}] 缺少 url 配置`);
      if (this.verbose) console.log(`  🌐 [${this.name}] 连接: ${url}`);

      this.transport = new MCPHttpTransport(url, this.config.headers || {}, { timeout: this.config.timeout || 30000 });
      this._connectTime = Date.now();

      const initResult = await this.transport.request(MCPProtocol.Methods.INITIALIZE, {
        protocolVersion: MCPProtocol.PROTOCOL_VERSION,
        capabilities: { roots: { listChanged: true }, sampling: {} },
        clientInfo: { name: 'pi-mvp', version: '0.4.0' }
      }, 15000);

      this.serverInfo = initResult.serverInfo;
      this.capabilities = initResult.capabilities || {};
      if (this.verbose) console.log(`  ✅ [${this.name}] 已初始化 (${initResult.serverInfo?.name || 'unknown'})`);

      await this.transport.notify(MCPProtocol.Methods.INITIALIZED_NOTIFICATION);
      await this._discoverAll();

      this._initialized = true;
      return this;
    } catch (err) {
      this._cleanup();
      throw new Error(`[${this.name}] 连接失败: ${err.message}`);
    }
  }

  async _discoverAll() {
    try {
      const toolsResult = await this.transport.request(MCPProtocol.Methods.LIST_TOOLS, {}, 10000);
      this.tools = (toolsResult.tools || []).map(t => ({ ...t, _mcpServer: this.name }));
      if (this.verbose) console.log(`     🔧 发现 ${this.tools.length} 个工具: ${this.tools.map(t => t.name).join(', ')}`);
    } catch (err) { if (this.verbose) console.log(`     ⚠️  [${this.name}] 工具发现失败: ${err.message}`); }

    try {
      const resourcesResult = await this.transport.request(MCPProtocol.Methods.LIST_RESOURCES, {}, 5000);
      this.resources = resourcesResult.resources || [];
    } catch { /* 可选 */ }

    try {
      const promptsResult = await this.transport.request(MCPProtocol.Methods.LIST_PROMPTS, {}, 5000);
      this.prompts = promptsResult.prompts || [];
    } catch { /* 可选 */ }
  }

  async callTool(toolName, args = {}) {
    if (!this.isReady) throw new Error(`[${this.name}] 未初始化或已断开`);
    this._callCount++;
    if (this.verbose) { const p = JSON.stringify(args).slice(0, 150); console.log(`  🔧 [${this.name}.${toolName}](${p}${p.length >= 150 ? '...' : ''})`); }

    try {
      const result = await this.transport.request(MCPProtocol.Methods.CALL_TOOL, { name: toolName, arguments: args }, 120000);
      const output = { success: !result.isError, content: result.content || [], isError: result.isError || false };
      if (this.verbose) { const p = JSON.stringify(result.content).slice(0, 200); console.log(`  ✅ [${this.name}.${toolName}] ← ${p}${p.length >= 200 ? '...' : ''}`); }
      return output;
    } catch (err) {
      if (this.verbose) console.log(`  ❌ [${this.name}.${toolName}] 错误: ${err.message}`);
      return { success: false, error: err.message, isError: true };
    }
  }

  async readResource(uri) {
    if (!this.isReady) throw new Error(`[${this.name}] 未初始化`);
    return (await this.transport.request(MCPProtocol.Methods.READ_RESOURCE, { uri }, 30000)).contents || [];
  }

  async getPrompt(name, args = {}) {
    if (!this.isReady) throw new Error(`[${this.name}] 未初始化`);
    return await this.transport.request(MCPProtocol.Methods.GET_PROMPT, { name, arguments: args }, 10000);
  }

  async ping() {
    if (!this.transport) return false;
    try { await this.transport.request(MCPProtocol.Methods.PING, {}, 5000); return true; } catch { return false; }
  }

  disconnect() {
    if (this.verbose) console.log(`  🔌 [${this.name}] 断开 (调用次数: ${this._callCount})`);
    this._cleanup();
  }

  _cleanup() {
    if (this.transport) { this.transport.close(); this.transport = null; }
    this._initialized = false;
    this._connectTime = null;
  }

  toJSON() {
    return { name: this.name, url: this.config.url, ready: this.isReady, serverInfo: this.serverInfo, toolCount: this.tools.length, resourceCount: this.resources.length, promptCount: this.prompts.length, callCount: this._callCount, uptime: this.uptime };
  }
}

class MCPClientManager {
  constructor(options = {}) {
    this.servers = new Map();
    this.configPath = options.configPath || Utils.configPath('mcp-servers.json');
    this.verbose = options.verbose ?? true;
    this._globalTools = [];
    this._dirty = false;
  }

  get serverCount() { return Array.from(this.servers.values()).filter(s => s.isReady).length; }
  get totalToolCount() { return this.getAllTools().length; }

  loadConfig() {
    let config = Utils.readJSON(this.configPath, null);
    if (!config || !config.mcpServers) {
      const mainConfig = Utils.readJSON(Utils.configPath('config.json'), {});
      config = mainConfig.mcpServers ? { mcpServers: mainConfig.mcpServers } : {};
    }
    return config.mcpServers || {};
  }

  saveConfig(serversConfig) { Utils.writeJSON(this.configPath, { mcpServers: serversConfig }); return this.configPath; }

  async connectServer(name, config) {
    if (this.servers.has(name)) { const existing = this.servers.get(name); if (existing.isReady) return existing; existing.disconnect(); }
    const server = new MCPServer(name, config, { verbose: this.verbose });
    this.servers.set(name, server);
    await server.connect();
    this._dirty = true;
    return server;
  }

  async connectAll(serverConfigs = null) {
    const configs = serverConfigs || this.loadConfig();
    const names = Object.keys(configs);
    if (names.length === 0) { if (this.verbose) console.log('\n📦 MCP: 无已配置的 Server'); return []; }
    if (this.verbose) console.log(`\n📦 MCP: 正在连接 ${names.length} 个 HTTP Server...`);

    const results = [], errors = [];
    await Promise.allSettled(names.map(async (name) => {
      try {
        const server = await this.connectServer(name, configs[name]);
        results.push({ name, status: 'ok', tools: server.tools.length });
      } catch (err) {
        errors.push({ name, error: err.message });
        if (this.verbose) console.log(`  ❌ [${name}] ${err.message}`);
      }
    }));

    if (this.verbose && results.length > 0) console.log(`  ✅ 成功: ${results.length}/${names.length} | 总工具数: ${this.totalToolCount}`);
    if (errors.length > 0) console.log(`  ⚠️  失败: ${errors.map(e => `[${e.name}] ${e.error}`).join('; ')}`);
    return results;
  }

  disconnectServer(name) { const server = this.servers.get(name); if (server) { server.disconnect(); this.servers.delete(name); this._dirty = true; } }
  disconnectAll() { for (const [, server] of this.servers) server.disconnect(); this.servers.clear(); this._dirty = true; if (this.verbose) console.log('\n📦 MCP: 所有 Server 已断开'); }

  getAllTools() {
    if (!this._dirty && this._globalTools.length > 0) return this._globalTools;
    const allTools = [];
    for (const [serverName, server] of this.servers) {
      if (server.isReady) for (const tool of server.tools) allTools.push({ ...tool, _mcpServer: serverName });
    }
    this._globalTools = allTools;
    this._dirty = false;
    return allTools;
  }

  async routeCall(toolName, args = {}) {
    for (const [, server] of this.servers) {
      if (!server.isReady) continue;
      if (server.tools.some(t => t.name === toolName)) return server.callTool(toolName, args);
    }
    throw new Error(`MCP 工具未找到: ${toolName} (已连接 ${this.serverCount} 个 Server)`);
  }

  findServerForTool(toolName) {
    for (const [serverName, server] of this.servers) {
      if (server.isReady && server.tools.some(t => t.name === toolName)) return { serverName, server };
    }
    return null;
  }

  getStatus() {
    return { totalServers: this.servers.size, connectedServers: this.serverCount, totalTools: this.totalToolCount, servers: Array.from(this.servers.values()).map(s => s.toJSON()) };
  }

  async restartServer(name) {
    const server = this.servers.get(name);
    const config = server?.config;
    if (!config) throw new Error(`未知 Server: ${name}`);
    this.disconnectServer(name);
    return this.connectServer(name, config);
  }
}

// ============================================================
// 4. Skills 系统 — 按需加载，意图驱动
// ============================================================

/**
 * Skill 定义结构（存储在 workspace/skills/<skill-name>/SKILL.md）
 *
 * ---
 * name: skill-name
 * description: 一句话描述这个技能的用途（用于意图匹配）
 * version: 1.0.0
 * triggers:
 *   - "关键词1"
 *   - "关键词2"
 * category: coding|search|analysis|creative|system|other
 * requires_tools:
 *   - read_file
 *   - write_file
 * llm_role: planner | profile | chat | router  # 可选，指定使用哪个 LLM 角色
 * ---
 *
 * # 技能详细说明
 *
 * （以下内容仅在技能被激活后才加载到上下文）
 */

class Skill {
  /**
   * @param {object} meta - 技能元数据
   * @param {string} meta.name - 技能名称（目录名）
   * @param {string} meta.description - 极简描述（用于意图匹配）
   * @param {string} meta.version - 版本号
   * @param {string[]} meta.triggers - 触发关键词列表
   * @param {string} meta.category - 分类
   * @param {string[]} meta.requires_tools - 需要的工具
   * @param {string} meta.llm_role - 推荐使用的 LLM 角色
   * @param {string} content - 技能完整内容（SKILL.md 的 body 部分）
   * @param {string} skillDir - 技能目录绝对路径
   */
  constructor(meta, content, skillDir) {
    this.name = meta.name;
    this.description = meta.description || '';
    this.version = meta.version || '1.0.0';
    this.triggers = meta.triggers || [];
    this.category = meta.category || 'other';
    this.requiresTools = meta.requires_tools || [];
    this.llmRole = meta.llm_role || null;
    this.content = content || '';
    this.skillDir = skillDir;
    this._loaded = false;  // 是否已完整加载内容
  }

  /** 匹配用户输入是否触发此技能 */
  match(input) {
    if (!input || !this.triggers.length) return false;
    const lowerInput = input.toLowerCase();
    return this.triggers.some(trigger => lowerInput.includes(trigger.toLowerCase()));
  }

  /** 获取轻量级摘要（用于技能列表展示） */
  get summary() {
    return {
      name: this.name,
      description: this.description,
      category: this.category,
      triggerCount: this.triggers.length,
      llmRole: this.llmRole,
      loaded: this._loaded
    };
  }

  /** 加载完整内容 */
  load() {
    this._loaded = true;
    return this.content;
  }

  toJSON() {
    return this.summary;
  }
}

/**
 * SkillManager — 技能管理器
 *
 * 核心设计原则：
 * 1. 启动时只扫描 skills 目录，读取每个技能的 frontmatter（元数据）
 * 2. 不加载技能正文内容，只保留 name/description/triggers 等元信息
 * 3. 收到用户输入后，先进行意图路由分析，判断是否需要激活某个技能
 * 4. 仅在被激活时才加载完整的技能内容到上下文
 */
class SkillManager {
  constructor(options = {}) {
    this.skillsDir = options.skillsDir || SKILLS_DIR;
    this.skills = new Map();  // name -> Skill 实例
    this.router = options.router || null;  // LLMRouter 实例（用于意图路由）
    this.verbose = options.verbose ?? true;
    this._scanned = false;
  }

  /** 扫描 skills 目录，仅加载元数据（frontmatter） */
  scan() {
    if (!fs.existsSync(this.skillsDir)) {
      Utils.ensureDir(this.skillsDir);
      if (this.verbose) console.log(`  📁 Skills 目录已创建: ${this.skillsDir}`);
      this._scanned = true;
      return this;
    }

    const entries = fs.readdirSync(this.skillsDir, { withFileTypes: true });
    let loaded = 0;

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const skillDir = path.join(this.skillsDir, entry.name);
      const skillFile = path.join(skillDir, 'SKILL.md');

      if (!fs.existsSync(skillFile)) continue;

      try {
        const skill = this._parseSkill(entry.name, skillFile);
        if (skill) {
          this.skills.set(skill.name, skill);
          loaded++;
        }
      } catch (err) {
        if (this.verbose) console.log(`  ⚠️  技能解析失败 [${entry.name}]: ${err.message}`);
      }
    }

    this._scanned = true;
    if (this.verbose) console.log(`  🎯 Skills: 已发现 ${loaded} 个技能（仅元数据）`);
    return this;
  }

  /** 解析单个 SKILL.md 文件，只提取 frontmatter 元数据 */
  _parseSkill(name, skillFilePath) {
    const raw = fs.readFileSync(skillFilePath, 'utf-8');
    const frontmatterMatch = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);

    if (!frontmatterMatch) {
      console.log(`  ⚠️  [${name}] SKILL.md 格式错误: 缺少 frontmatter`);
      return null;
    }

    const fmRaw = frontmatterMatch[1];
    const content = frontmatterMatch[2].trim();

    // 解析简单的 YAML-like frontmatter
    const meta = { name };
    const lines = fmRaw.split('\n');
    for (const line of lines) {
      const colonIdx = line.indexOf(':');
      if (colonIdx <= 0) continue;
      const key = line.slice(0, colonIdx).trim();
      let value = line.slice(colonIdx + 1).trim();

      // 处理数组格式（以 - 开头的行在后续处理）
      if (key === 'triggers' || key === 'requires_tools') {
        meta[key.replace('-', '_')] = [];
        continue;  // 数组在下一轮处理
      }

      // 处理字符串值（去除引号）
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      meta[key] = value;
    }

    // 后处理数组字段
    let currentArray = null;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('- ')) {
        const val = trimmed.slice(2).trim().replace(/^["']|["']$/g, '');
        if (currentArray) meta[currentArray].push(val);
      } else if (line.includes(':')) {
        currentArray = null;
        const key = line.slice(0, line.indexOf(':')).trim();
        if (key === 'triggers') currentArray = 'triggers';
        else if (key === 'requires_tools') currentArray = 'requires_tools';
      }
    }

    // 确保 arrays 存在
    if (!meta.triggers) meta.triggers = [];
    if (!meta.requires_tools) meta.requires_tools = [];

    return new Skill(meta, content, path.dirname(skillFilePath));
  }

  /** 获取所有技能的轻量级摘要列表 */
  list() {
    return Array.from(this.skills.values()).map(s => s.summary);
  }

  /** 按名称获取技能（懒加载） */
  get(name) {
    return this.skills.get(name) || null;
  }

  /**
   * 意图路由 — 分析用户输入，决定是否激活某个技能
   *
   * 流程：
   * 1. 关键词精确匹配（零成本）
   * 2. 如果有 LLM Router，使用 router 模型进行语义分析
   * 3. 返回最佳匹配的技能（或 null 表示无需技能）
   */
  async route(userInput) {
    if (!userInput || this.skills.size === 0) return null;

    // 第一层：关键词精确匹配
    const matches = [];
    for (const [, skill] of this.skills) {
      if (skill.match(userInput)) {
        matches.push({ skill, score: 1.0, reason: 'keyword-match' });
      }
    }

    if (matches.length === 1) {
      const best = matches[0].skill;
      best.load();
      if (this.verbose) console.log(`  🎯 技能激活 [关键词]: ${best.name}`);
      return best;
    }

    if (matches.length > 1) {
      // 多个匹配，选择第一个
      const best = matches[0].skill;
      best.load();
      if (this.verbose) console.log(`  🎯 技能激活 [多选]: ${best.name} (共 ${matches.length} 个候选)`);
      return best;
    }

    // 第二层：LLM 语义路由（如果有 router Provider）
    if (this.router) {
      try {
        const routed = await this._llmRoute(userInput);
        if (routed) return routed;
      } catch (err) {
        if (this.verbose) console.log(`  ⚠️  LLM 路由失败: ${err.message}`);
      }
    }

    return null;
  }

  /** 使用 LLM 进行意图路由 */
  async _llmRoute(userInput) {
    const routerProvider = this.router.router;
    const skillList = this.list().map(s => `- ${s.name}: ${s.description}`).join('\n');

    const messages = [
      {
        role: 'system',
        content: `你是意图路由器。根据用户输入和可用技能列表，判断是否需要激活某个技能。

可用技能:
${skillList}

规则:
- 如果用户请求明确匹配某个技能的功能，返回该技能的 name
- 如果没有匹配的技能，返回 "none"
- 只返回一个 JSON 对象: {"skill": "技能名或none", "confidence": 0.0-1.0}`
      },
      { role: 'user', content: userInput }
    ];

    const response = await routerProvider.chat(messages, { temperature: 0.1, maxTokens: 100 });
    const text = (response.content || '').trim();

    try {
      const parsed = JSON.parse(text);
      if (parsed.skill && parsed.skill !== 'none' && parsed.confidence > 0.5) {
        const skill = this.skills.get(parsed.skill);
        if (skill) {
          skill.load();
          if (this.verbose) console.log(`  🎯 技能激活 [LLM]: ${skill.name} (置信度: ${parsed.confidence})`);
          return skill;
        }
      }
    } catch { /* JSON parse fail */ }

    return null;
  }

  /** 创建新技能 */
  create(name, options = {}) {
    const skillDir = path.join(this.skillsDir, name);
    if (fs.existsSync(skillDir)) {
      throw new Error(`技能已存在: ${name}`);
    }

    Utils.ensureDir(skillDir);

    const skillContent = `---
name: ${name}
description: ${options.description || '请补充技能描述'}
version: 1.0.0
triggers:
  - "${options.trigger || name}"
category: ${options.category || 'other'}
requires_tools: []
---

# ${name}

## 概述

${options.description || '请描述这个技能的用途'}

## 能力

<!-- 描述这个技能提供的具体能力 -->

## 使用场景

<!-- 描述何时应该使用这个技能 -->

## 指导原则

<!-- 给 AI 的指导原则 -->
`;

    fs.writeFileSync(path.join(skillDir, 'SKILL.md'), skillContent, 'utf-8');

    // 重新扫描
    this.scan();

    if (this.verbose) console.log(`✅ 技能已创建: ${name} → ${skillDir}`);
    return this.get(name);
  }

  /** 删除技能 */
  remove(name) {
    const skill = this.skills.get(name);
    if (!skill) throw new Error(`技能不存在: ${name}`);

    // 删除目录
    fs.rmSync(skill.skillDir, { recursive: true, force: true });
    this.skills.delete(name);

    if (this.verbose) console.log(`🗑️  技能已删除: ${name}`);
    return true;
  }

  get count() { return this.skills.size; }
  get scanned() { return this._scanned; }
}

// ============================================================
// 5. 工具系统 (Tool System) — 含 MCP 桥接
// ============================================================

class Tool {
  constructor(def) {
    this.name = def.name;
    this.description = def.description;
    this.parameters = def.parameters || {};
    this.execute = def.execute;
    this.source = def.source || 'builtin';
  }

  toOpenAIFunc() {
    return { type: 'function', function: { name: this.name, description: this.description, parameters: this.parameters } };
  }

  toAnthropicTool() {
    return { name: this.name, description: this.description, input_schema: this.parameters };
  }
}

/** 内置工具集 */
const BuiltinTools = {
  read_file: new Tool({
    name: 'read_file',
    description: '读取文件内容，返回完整文本。',
    parameters: { type: 'object', properties: { filePath: { type: 'string', description: '文件路径' }, encoding: { type: 'string', default: 'utf-8' } }, required: ['filePath'] },
    async execute({ filePath, encoding = 'utf-8' }, context) {
      const resolved = path.isAbsolute(filePath) ? filePath : path.join(context.workDir || process.cwd(), filePath);
      if (!fs.existsSync(resolved)) return { error: `文件不存在: ${resolved}` };
      const stat = fs.statSync(resolved);
      if (stat.size > 1024 * 500) return { error: `文件过大 (${(stat.size / 1024).toFixed(0)}KB)` };
      const content = fs.readFileSync(resolved, encoding);
      return { success: true, path: resolved, content, size: stat.size, lines: content.split('\n').length };
    }
  }),

  write_file: new Tool({
    name: 'write_file',
    description: '写入文件，自动创建父目录。',
    parameters: { type: 'object', properties: { filePath: { type: 'string', description: '目标路径' }, content: { type: 'string', description: '内容' }, encoding: { type: 'string', default: 'utf-8' } }, required: ['filePath', 'content'] },
    async execute({ filePath, content, encoding = 'utf-8' }, context) {
      const resolved = path.isAbsolute(filePath) ? filePath : path.join(context.workDir || process.cwd(), filePath);
      Utils.ensureDir(path.dirname(resolved));
      fs.writeFileSync(resolved, content, encoding);
      return { success: true, path: resolved, size: Buffer.byteLength(content, encoding), message: `已写入 ${resolved}` };
    }
  }),

  list_dir: new Tool({
    name: 'list_dir',
    description: '列出目录内容。',
    parameters: { type: 'object', properties: { dirPath: { type: 'string', description: '目录路径' }, recursive: { type: 'boolean', default: false }, pattern: { type: 'string', default: '*' } }, required: [] },
    async execute({ dirPath, recursive = false, pattern = '*' }, context) {
      const baseDir = dirPath ? (path.isAbsolute(dirPath) ? dirPath : path.join(context.workDir || process.cwd(), dirPath)) : (context.workDir || process.cwd());
      if (!fs.existsSync(baseDir)) return { error: `目录不存在: ${baseDir}` };
      const items = [];
      const scan = (dir, depth = 0) => {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          const fullPath = path.join(dir, entry.name);
          const relPath = path.relative(baseDir, fullPath);
          if (pattern !== '*') { const regex = new RegExp('^' + pattern.replace(/\*/g, '.*').replace(/\?/g, '.') + '$'); if (!regex.test(entry.name)) continue; }
          items.push({ name: entry.name, path: relPath || entry.name, type: entry.isDirectory() ? 'directory' : 'file', depth });
          if (entry.isDirectory() && recursive && depth < 10) scan(fullPath, depth + 1);
        }
      };
      scan(baseDir);
      return { success: true, path: baseDir, count: items.length, items };
    }
  }),

  search_files: new Tool({
    name: 'search_files',
    description: '搜索包含特定文本或正则表达式的文件。',
    parameters: { type: 'object', properties: { query: { type: 'string', description: '搜索关键词或正则' }, dirPath: { type: 'string', description: '搜索目录' }, filePattern: { type: 'string', default: '*' }, maxResults: { type: 'number', default: 20 } }, required: ['query'] },
    async execute({ query, dirPath, filePattern = '*', maxResults = 20 }, context) {
      const baseDir = dirPath ? (path.isAbsolute(dirPath) ? dirPath : path.join(context.workDir || process.cwd(), dirPath)) : (context.workDir || process.cwd());
      const results = [];
      const regex = new RegExp(query, 'i');
      const walk = (dir) => {
        if (!fs.existsSync(dir)) return;
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) { walk(fullPath); continue; }
          if (filePattern !== '*' && !new RegExp('^' + filePattern.replace(/\*/g, '.*') + '$').test(entry.name)) continue;
          try {
            const lines = fs.readFileSync(fullPath, 'utf-8').split('\n');
            for (let i = 0; i < lines.length && results.length < maxResults; i++) {
              if (regex.test(lines[i])) results.push({ file: path.relative(baseDir, fullPath), line: i + 1, match: lines[i].trim().slice(0, 200) });
            }
          } catch { /* skip binary */ }
        }
      };
      walk(baseDir);
      return { success: true, query, total: results.length, results };
    }
  }),

  file_info: new Tool({
    name: 'file_info',
    description: '获取文件的元信息。',
    parameters: { type: 'object', properties: { filePath: { type: 'string', description: '文件或目录路径' } }, required: ['filePath'] },
    async execute({ filePath }, context) {
      const resolved = path.isAbsolute(filePath) ? filePath : path.join(context.workDir || process.cwd(), filePath);
      if (!fs.existsSync(resolved)) return { error: `不存在: ${resolved}` };
      const stat = fs.statSync(resolved);
      return { success: true, path: resolved, isDirectory: stat.isDirectory(), isFile: stat.isFile(), size: stat.size, created: stat.birthtime.toISOString(), modified: stat.mtime.toISOString() };
    }
  }),

  think: new Tool({
    name: 'think',
    description: '内部思考和规划，无副作用。',
    parameters: { type: 'object', properties: { thought: { type: 'string', description: '思考内容' } }, required: ['thought'] },
    async execute({ thought }) { return { success: true, thought, timestamp: new Date().toISOString() }; }
  }),

  // ====== MCP 管理内置工具 ======

  mcp_list_servers: new Tool({
    name: 'mcp_list_servers',
    description: '列出所有 MCP Server 及其状态、可用工具数量。',
    parameters: { type: 'object', properties: {}, required: [] },
    async execute(params, context) {
      const mgr = context.mcpManager;
      if (!mgr) return { error: 'MCP Manager 未初始化' };
      const status = mgr.getStatus();
      return { success: true, summary: { totalServers: status.totalServers, connectedServers: status.connectedServers, totalTools: status.totalTools }, servers: status.servers.map(s => ({ name: s.name, url: s.url, ready: s.ready, toolCount: s.toolCount, callCount: s.callCount })) };
    }
  }),

  mcp_list_tools: new Tool({
    name: 'mcp_list_tools',
    description: '列出所有 MCP Server 提供的工具详情。',
    parameters: { type: 'object', properties: { serverName: { type: 'string', description: '按 Server 过滤' } }, required: [] },
    async execute({ serverName }, context) {
      const mgr = context.mcpManager;
      if (!mgr) return { error: 'MCP Manager 未初始化' };
      let tools = mgr.getAllTools();
      if (serverName) tools = tools.filter(t => t._mcpServer === serverName);
      return { success: true, count: tools.length, tools: tools.map(t => ({ name: t.name, description: t.description?.slice(0, 200), parameters: Object.keys(t.inputSchema?.properties || {}), server: t._mcpServer })) };
    }
  }),

  mcp_restart_server: new Tool({
    name: 'mcp_restart_server',
    description: '重启指定的 MCP Server。',
    parameters: { type: 'object', properties: { serverName: { type: 'string', description: 'Server 名称' } }, required: ['serverName'] },
    async execute({ serverName }, context) {
      const mgr = context.mcpManager;
      if (!mgr) return { error: 'MCP Manager 未初始化' };
      try { const server = await mgr.restartServer(serverName); return { success: true, message: `Server ${serverName} 已重启`, tools: server.tools.length }; }
      catch (err) { return { error: err.message }; }
    }
  }),

  // ====== Skills 管理内置工具 ======

  skill_list: new Tool({
    name: 'skill_list',
    description: '列出所有可用技能及其简要描述（不含详细内容）。',
    parameters: { type: 'object', properties: { category: { type: 'string', description: '按分类过滤' } }, required: [] },
    async execute({ category }, context) {
      const mgr = context.skillManager;
      if (!mgr) return { error: 'Skill Manager 未初始化' };
      let skills = mgr.list();
      if (category) skills = skills.filter(s => s.category === category);
      return { success: true, count: skills.length, skills };
    }
  }),

  skill_load: new Tool({
    name: 'skill_load',
    description: '按名称加载并激活一个技能，返回完整内容。',
    parameters: { type: 'object', properties: { name: { type: 'string', description: '技能名称' } }, required: ['name'] },
    async execute({ name }, context) {
      const mgr = context.skillManager;
      if (!mgr) return { error: 'Skill Manager 未初始化' };
      const skill = mgr.get(name);
      if (!skill) return { error: `技能不存在: ${name}` };
      skill.load();
      return { success: true, name: skill.name, description: skill.description, content: skill.content, category: skill.category };
    }
  })
};

function getAllBuiltinTools() { return Object.values(BuiltinTools); }
function getBuiltinToolByName(name) { return BuiltinTools[name] || null; }

function mcpToolToAgentTool(mcpToolDef, mcpManager) {
  return new Tool({
    name: mcpToolDef.name,
    description: `[MCP:${mcpToolDef._mcpServer}] ${mcpToolDef.description || ''}`,
    parameters: mcpToolDef.inputSchema || {},
    source: `mcp:${mcpToolDef._mcpServer}`,
    execute: async (args) => mcpManager.routeCall(mcpToolDef.name, args)
  });
}

function buildMergedTools(mcpManager) {
  const tools = [...getAllBuiltinTools()];
  if (mcpManager) {
    for (const t of mcpManager.getAllTools()) tools.push(mcpToolToAgentTool(t, mcpManager));
  }
  return tools;
}

function findToolByName(name, mcpManager) {
  const builtin = getBuiltinToolByName(name);
  if (builtin) return builtin;
  if (mcpManager) {
    const location = mcpManager.findServerForTool(name);
    if (location) {
      const def = location.server.tools.find(t => t.name === name);
      if (def) return mcpToolToAgentTool(def, mcpManager);
    }
  }
  return null;
}

// ============================================================
// 6. Agent 运行时 (Agent Runtime) — 含 Skills 集成
// ============================================================

class Session {
  constructor(options = {}) {
    this.id = options.id || Utils.uid();
    this.createdAt = new Date().toISOString();
    this.messages = [];
    this.state = {};
    this.metadata = options.metadata || {};
    this.workDir = options.workDir || process.cwd();
    this.toolResults = [];
    this.roundCount = 0;
    this.activeSkill = null;  // 当前激活的技能
  }

  addMessage(role, content) {
    this.messages.push({ role, content: typeof content === 'string' ? content : JSON.stringify(content), timestamp: new Date().toISOString() });
  }

  getHistory() { return this.messages.map(m => ({ role: m.role, content: m.content })); }
  setState(key, value) { this.state[key] = value; }

  activateSkill(skill) {
    this.activeSkill = skill ? { name: skill.name, description: skill.description, activatedAt: new Date().toISOString() } : null;
  }

  toJSON() {
    return {
      id: this.id, createdAt: this.createdAt, messageCount: this.messages.length,
      state: this.state, metadata: this.metadata, workDir: this.workDir,
      toolResultCount: this.toolResults.length, activeSkill: this.activeSkill
    };
  }
}

class Agent {
  constructor(config = {}) {
    this.llmRouter = config.llmRouter;           // LLMRouter 实例（多 Provider 路由）
    this.provider = config.provider;              // 兼容旧版：直接传入的 Provider
    this.mcpManager = config.mcpManager || null;
    this.skillManager = config.skillManager || null;  // SkillManager 实例
    this.systemPrompt = config.systemPrompt || DEFAULT_SYSTEM_PROMPT;
    this.session = config.session || new Session({ workDir: config.workDir });
    this.verbose = config.verbose ?? true;
    this.maxIterations = config.maxIterations || 30;
    this.tools = config.tools || buildMergedTools(this.mcpManager);
  }

  setSystemPrompt(prompt) { this.systemPrompt = prompt; return this; }
  refreshTools() { this.tools = buildMergedTools(this.mcpManager); return this.tools.length; }

  /** 获取当前对话用的 Provider（优先使用技能指定的 llmRole） */
  _getProvider(activeSkill) {
    // 如果有技能且指定了 llmRole，使用路由器获取对应 Provider
    if (activeSkill && activeSkill.llmRole && this.llmRouter) {
      return this.llmRouter.get(activeSkill.llmRole);
    }
    // 否则用默认 Provider 或路由器的 chat
    return this.provider || (this.llmRouter ? this.llmRouter.chat : null);
  }

  _buildDynamicSystemPrompt(activeSkill) {
    const builtinNames = getAllBuiltinTools().map(t => t.name).filter(n => !n.startsWith('mcp_') && !n.startsWith('skill_'));

    let prompt = this.systemPrompt;

    // 注入技能内容（如果已激活）
    if (activeSkill) {
      const skillSection = `
## 当前激活技能: ${activeSkill.name}
> ${activeSkill.description}

### 技能指导
${activeSkill.content}
`;
      prompt = prompt + '\n' + skillSection;
    }

    // 注入 MCP 工具列表
    if (this.mcpManager && this.mcpManager.totalToolCount > 0) {
      const sections = {};
      for (const tool of this.mcpManager.getAllTools()) {
        const svr = tool._mcpServer || 'unknown';
        if (!sections[svr]) sections[svr] = [];
        sections[svr].push(`  - ${tool.name}: ${(tool.description || '').replace(/\[MCP:[^\]]*\]\s*/, '')}`);
      }
      let mcpText = '\n## MCP 外部工具\n';
      for (const [svr, tools] of Object.entries(sections)) {
        mcpText += `\### ${svr}\n${tools.join('\n')}\n`;
      }
      prompt = prompt.replace(/## 可用工具[\s\S]*?(?=##|$)/, `## 可用工具\n${builtinNames.map(n => `- ${n}`).join('\n')}${mcpText}`);
    }

    return prompt;
  }

  async run(userInput) {
    // ========== 意图路由阶段 ==========
    let activeSkill = null;
    if (this.skillManager && this.skillManager.scanned) {
      try {
        activeSkill = await this.skillManager.route(userInput);
        if (activeSkill) {
          this.session.activateSkill(activeSkill);
          if (this.verbose) console.log(`\n🎯 技能已激活: ${activeSkill.name} (${activeSkill.category})`);
        }
      } catch (err) {
        if (this.verbose) console.log(`⚠️  技能路由跳过: ${err.message}`);
      }
    }

    // ========== 正常 Agent 循环 ==========
    this.session.addMessage('user', userInput);

    const provider = this._getProvider(activeSkill);
    if (!provider) throw new Error('无可用 LLM Provider');

    const messages = [
      { role: 'system', content: this._buildDynamicSystemPrompt(activeSkill) },
      ...this.session.getHistory()
    ];
    const tools = this._buildToolDefinitions();

    for (let i = 0; i < this.maxIterations; i++) {
      this.session.roundCount++;
      if (this.verbose) {
        console.log(`\n${'─'.repeat(60)}`);
        console.log(`🔄 Agent 第 ${this.session.roundCount} 轮` + (activeSkill ? ` [${activeSkill.name}]` : '') + ` | Provider: ${provider.name}/${provider.model}`);
        console.log(`${'─'.repeat(60)}`);
      }

      try {
        const response = await provider.chat(messages, { tools });

        if (response.toolCalls && response.toolCalls.length > 0) {
          this.session.addMessage('assistant', response.content || '(正在调用工具...)');

          for (const tc of response.toolCalls) {
            const toolName = tc.name;
            const args = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : tc.arguments;
            if (this.verbose) console.log(`  🔧 工具调用: ${toolName}(${JSON.stringify(args).slice(0, 100)})`);

            const tool = findToolByName(toolName, this.mcpManager);
            if (!tool) {
              const errMsg = `未知工具: ${toolName}`;
              if (this.verbose) console.log(`  ❌ ${errMsg}`);
              this._appendToolResult(toolName, args, { error: errMsg }, messages);
              continue;
            }

            try {
              const result = await tool.execute(args, {
                workDir: this.session.workDir, session: this.session,
                mcpManager: this.mcpManager, skillManager: this.skillManager
              });
              if (this.verbose) { const p = JSON.stringify(result).slice(0, 200); console.log(`  ✅ 结果: ${p}${p.length > 200 ? '...' : ''}`); }
              this.session.toolResults.push({ toolName, args, result, round: this.session.roundCount, source: tool.source });
              this._appendToolResult(toolName, args, result, messages);
            } catch (err) {
              if (this.verbose) console.log(`  ❌ 错误: ${err.message}`);
              this._appendToolResult(toolName, args, { error: err.message, stack: err.stack }, messages);
            }
          }
        } else {
          this.session.addMessage('assistant', response.content);
          return {
            success: true, content: response.content, rounds: this.session.roundCount,
            toolCalls: this.session.toolResults.length,
            mcpCalls: this.session.toolResults.filter(t => t.source?.startsWith('mcp:')).length,
            activeSkill: activeSkill ? activeSkill.name : null,
            session: this.session.toJSON()
          };
        }
      } catch (err) {
        if (this.verbose) console.log(`  💥 异常: ${err.message}`);
        if (i >= 2) {
          return { success: false, error: err.message, partial: true, rounds: this.session.roundCount, session: this.session.toJSON() };
        }
        throw err;
      }
    }

    return { success: false, error: '达到最大迭代次数', rounds: this.session.roundCount, session: this.session.toJSON() };
  }

  _appendToolResult(toolName, args, result, messages) {
    messages.push({ role: 'tool', tool_call_id: `${toolName}-${Date.now()}`, name: toolName, content: JSON.stringify(result) });
  }

  _buildToolDefinitions() {
    const provider = this.provider || (this.llmRouter ? this.llmRouter.chat : null);
    if (provider instanceof AnthropicProvider) return this.tools.map(t => t.toAnthropicTool());
    return this.tools.map(t => t.toOpenAIFunc());
  }
}

// ============================================================
// 7. 默认系统提示词
// ============================================================

const DEFAULT_SYSTEM_PROMPT = `你是一个高效的 AI 编程助手（Pi-Agent），具备以下能力：

## 可用工具
- read_file: 读取文件内容
- write_file: 写入文件
- list_dir: 列出目录
- search_files: 搜索文件内容
- file_info: 获取文件信息
- think: 内部思考与规划
- skill_list: 列出可用技能
- skill_load: 加载指定技能
- mcp_list_servers: 查看 MCP Server 状态
- mcp_list_tools: 查看 MCP 工具列表
- mcp_restart_server: 重启 MCP Server

## 工作原则
1. **理解优先** — 先用 think 分析需求，再动手
2. **分步执行** — 复杂任务拆分为小步骤
3. **先读后写** — 修改前先用 read_file 了解现有代码
4. **确认变更** — 重要操作前说明计划
5. **简洁高效** — 直接给出可运行的代码

## MCP 工具使用
- 使用外部服务时优先查看可用的 MCP 工具
- MCP 工具调用方式与内置工具相同

## 技能系统
- 当用户请求匹配某个技能的能力时，自动激活该技能
- 可以主动用 skill_list 查看可用技能
- 用 skill_load 手动加载特定技能

## 输出规范
- 代码块标注语言类型
- 修改文件时说明改动原因
- 遇到错误时提供修复建议`;

// ============================================================
// 8. 配置管理 (Config) — 支持多 LLM + workspace
// ============================================================

/**
 * WorkspaceConfigManager — 分层配置管理器
 *
 * 配置优先级（高→低）：
 *   1. 命令行参数
 *   2. 环境变量
 *   3. workspace/config.json（项目级）
 *   4. ~/.pi/config.json（用户级）
 *   5. 内置默认值
 */
const ConfigManager = {
  defaults: {
    agent: { name: 'Pi-Agent', max_rounds: 10 },
    llm: { base_url: '', api_key: '', model: '', streaming: true },
    llm_router: { base_url: '', api_key: '', model: '', streaming: true },
    llm_planner: { base_url: '', api_key: '', model: '', streaming: true },
    llm_profile: { base_url: '', api_key: '', model: '', streaming: true },
    provider: 'openai', model: 'gpt-4o-mini', apiKey: '', baseUrl: '',
    workDir: process.cwd(), maxIterations: 30, verbose: true, mcpEnabled: true, skillsEnabled: true
  },

  /**
   * 加载并合并所有配置源
   */
  load(customOverrides = {}) {
    // 1. 项目级配置 (workspace/config.json)
    const workspaceConfig = Utils.readJSON(WORKSPACE_CONFIG_PATH, {});

    // 2. 用户级配置 (~/.pi/config.json)
    const userConfig = Utils.readJSON(Utils.configPath('config.json'), {});

    // 3. 环境变量
    const envConfig = this._loadEnvVars();

    // 合并顺序：defaults < user < workspace < env < overrides
    const merged = this._deepMerge(
      this.defaults,
      userConfig,
      workspaceConfig,
      envConfig,
      customOverrides
    );

    return merged;
  },

  _loadEnvVars() {
    const env = {};
    if (process.env.PI_API_KEY) env.apiKey = process.env.PI_API_KEY;
    if (process.env.PI_PROVIDER) env.provider = process.env.PI_PROVIDER;
    if (process.env.PI_MODEL) env.model = process.env.PI_MODEL;
    if (process.env.PI_BASE_URL) env.baseUrl = process.env.PI_BASE_URL;
    if (process.env.OPENAI_API_KEY && !env.apiKey) env.apiKey = process.env.OPENAI_API_KEY;
    if (process.env.ANTHROPIC_API_KEY && !env.apiKey) env.apiKey = process.env.ANTHROPIC_API_KEY;
    if (process.env.GOOGLE_API_KEY && !env.apiKey) env.apiKey = process.env.GOOGLE_API_KEY;
    if (process.env.PI_MCP_ENABLED) env.mcpEnabled = process.env.PI_MCP_ENABLED !== 'false';
    return env;
  },

  _deepMerge(...sources) {
    const result = {};
    for (const source of sources) {
      if (!source || typeof source !== 'object') continue;
      for (const key of Object.keys(source)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
          result[key] = this._deepMerge(result[key] || {}, source[key]);
        } else if (source[key] !== undefined) {
          result[key] = source[key];
        }
      }
    }
    return result;
  },

  save(config) { Utils.writeJSON(WORKSPACE_CONFIG_PATH, config); return WORKSPACE_CONFIG_PATH; },

  /**
   * 从合并后的配置创建 LLMRouter
   */
  createLLMRouter(config) {
    return new LLMRouter(config);
  },

  /**
   * 兼容旧版：创建单个 Provider（用于无路由场景）
   */
  createProvider(config) {
    switch (config.provider) {
      case 'anthropic': return new AnthropicProvider(config);
      case 'google': return new GoogleProvider(config);
      default: return new OpenAICompatible(config);
    }
  }
};

// ============================================================
// 9. CLI 入口
// ============================================================

function parseArgs(argv) {
  const args = {
    prompt: null, provider: null, model: null, apiKey: null, baseUrl: null,
    workDir: null, nonInteractive: false, help: false, version: false,
    saveConfig: false, listModels: false,
    mcpStatus: false, mcpListTools: false, noMcp: false,
    mcpAdd: null, mcpRemove: null, mcpConfig: null,
    // 新增 Skills 参数
    skillCreate: null, skillList: false, skillRemove: null, noSkills: false
  };

  let i = 2;
  while (i < argv.length) {
    const a = argv[i];
    if (a === '-h' || a === '--help') args.help = true;
    else if (a === '-v' || a === '--version') args.version = true;
    else if (a === '-p' || a === '--provider') args.provider = argv[++i];
    else if (a === '-m' || a === '--model') args.model = argv[++i];
    else if (a === '--api-key') args.apiKey = argv[++i];
    else if (a === '--base-url') args.baseUrl = argv[++i];
    else if (a === '-d' || a === '--work-dir') args.workDir = argv[++i];
    else if (a === '-n' || a === '--non-interactive') args.nonInteractive = true;
    else if (a === '--save-config') args.saveConfig = true;
    else if (a === '--list-models') args.listModels = true;
    else if (a === '--mcp-status') args.mcpStatus = true;
    else if (a === '--mcp-tools') args.mcpListTools = true;
    else if (a === '--no-mcp') args.noMcp = true;
    else if (a === '--mcp-add') { args.mcpAdd = { name: argv[++i], url: argv[++i] }; }
    else if (a === '--mcp-remove') args.mcpRemove = argv[++i];
    else if (a === '--mcp-config') args.mcpConfig = argv[++i];
    // Skills 参数
    else if (a === '--skill-create') args.skillCreate = argv[++i];
    else if (a === '--skill-list') args.skillList = true;
    else if (a === '--skill-remove') args.skillRemove = argv[++i];
    else if (a === '--no-skills') args.noSkills = true;
    else if (!a.startsWith('-')) args.prompt = a;
    i++;
  }
  return args;
}

function showHelp() {
  console.log(`
╔══════════════════════════════════════════════════════╗
║     Pi-MVP  AI Agent Toolkit v0.4                    ║
║     单文件版 · 仅依赖 fs / path / os                 ║
║  ✅ 多 LLM 路由 · ✅ Skills 按需加载 · ✅ MCP HTTP  ║
╚══════════════════════════════════════════════════════╝

用法:
  node pi-mvp.js [选项] [prompt]

模式:
  无参数          交互模式（REPL）
  "你的问题"      单次问答模式

多 LLM 配置 (workspace/config.json):
  llm:           默认对话模型
  llm_router:    意图路由模型（轻量快速）
  llm_planner:   任务规划模型（强推理）
  llm_profile:   用户画像模型

LLM 选项:
  -p, --provider <name>    LLM 提供商 (openai|anthropic|google)
  -m, --model <name>       模型名称
  --api-key <key>          API 密钥
  --base-url <url>         API 基础 URL（兼容 API 如 Ollama）
  -d, --work-dir <path>    工作目录

Skills 选项:
  --skill-create <name>    创建新技能
  --skill-list             列出所有技能
  --skill-remove <name>    删除技能
  --no-skills              禁用技能系统

MCP 选项 (HTTP 模式):
  --mcp-status             查看 MCP Server 连接状态
  --mcp-tools              列出所有 MCP 可用工具
  --no-mcp                 禁用 MCP
  --mcp-add <name> <url>   添加 MCP HTTP Server
  --mcp-remove <name>      移除 MCP Server
  --mcp-config <path>      指定 MCP 配置文件

通用选项:
  -n, --non-interactive    非交互模式
  --list-models            列出模型
  --save-config            保存配置
  -h, --help               帮助
  -v, --version            版本

环境变量:
  PI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
  PI_MCP_ENABLED

示例:
  # 基本使用
  node pi-mvp.js "用 TypeScript 写一个快排"

  # Skills 管理
  node pi-mvp.js --skill-create code-review
  node pi-mvp.js --skill-list
  node pi-mvp.js --skill-remove old-skill

  # MCP HTTP Server
  node pi-mvp.js --mcp-add my-server http://localhost:3000/mcp
  node pi-mvp.js --mcp-status

  # 本地 Ollama
  node pi-mvp.js --base-url http://localhost:11434/v1 -m llama3 "你好"

  # REPL 中
  /skills        /skill-load <name>
  /mcp-status    /mcp-tools    /mcp-restart <name>
  /llm-status    /routes

配置文件:
  workspace/config.json       主配置（多 LLM + Agent 设置）
  ~/.pi/config.json           用户级配置
  ~/.pi/mcp-servers.json      MCP Server (URL-based)
  workspace/skills/           技能存储目录

Workspace 配置示例 (workspace/config.json):
{
  "agent": { "name": "AgentBei", "max_rounds": 10 },
  "llm": {
    "base_url": "https://api.deepseek.com",
    "api_key": "sk-xxx",
    "model": "deepseek-chat",
    "streaming": true
  },
  "llm_router": {
    "base_url": "https://api.deepseek.com",
    "api_key": "sk-xxx",
    "model": "deepseek-chat",
    "streaming": true
  },
  "llm_planner": { ... },
  "llm_profile": { ... }
}
`);
}

function showVersion() {
  console.log('Pi-MVP v0.4.0 — Single-file AI Agent Toolkit + MCP (HTTP) + Skills');
  console.log('依赖: fs, path, os (全部内置)');
  console.log('特性: 多 LLM 路由 | Skills 按需加载 | MCP Streamable HTTP');
}

function listModels(provider) {
  const p = ConfigManager.createProvider({ ...ConfigManager.defaults, ...{ provider: provider || 'openai' } });
  console.log(`\n📋 ${p.name} 可用模型:\n`);
  p.listModels().forEach((m, i) => console.log(`  ${i + 1}. ${m}`));
}

async function startRepl(agent, mcpManager, skillManager, llmRouter) {
  const readline = require('readline');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log('\n🚀 Pi-MVP v0.4 交互模式启动！');
  console.log('   输入问题开始对话');
  console.log('   /exit 退出 | /clear 清空 | /status 状态');
  if (skillManager) console.log('   /skills | /skill-load <name>');
  if (mcpManager) console.log('   /mcp-status | /mcp-tools | /mcp-restart <name>');
  if (llmRouter) console.log('   /llm-status | /routes');
  console.log('');

  const prompt = () => new Promise(resolve => rl.question('❯ ', resolve));

  while (true) {
    try {
      const input = await prompt();
      if (!input.trim()) continue;

      if (input === '/exit' || input === '/quit') { console.log('👋 再见！'); break; }
      if (input === '/clear') { agent.session = new Session({ workDir: agent.session.workDir }); console.log('✅ 已清空\n'); continue; }
      if (input === '/status') { console.log(JSON.stringify(agent.session.toJSON(), null, 2)); continue; }
      if (input.startsWith('/help')) { showHelp(); continue; }

      // Skills 命令
      if (input === '/skills' && skillManager) {
        const skills = skillManager.list();
        console.log(`\n🎯 可用技能 (${skills.length} 个):\n`);
        for (const s of skills) {
          console.log(`  [${s.category}] ${s.name}: ${s.description}`);
          console.log(`    触发词: ${s.triggerCount} 个 | ${s.loaded ? '✅已加载' : '⏳按需'}\n`);
        }
        continue;
      }
      if (input.startsWith('/skill-load ') && skillManager) {
        const name = input.slice('/skill-load '.length).trim();
        const skill = skillManager.get(name);
        if (!skill) { console.log(`❌ 技能不存在: ${name}\n`); continue; }
        skill.load();
        console.log(`✅ 技能已加载:\n  名称: ${skill.name}\n  描述: ${skill.description}\n  内容长度: ${skill.content.length} 字符\n`);
        continue;
      }

      // MCP 命令
      if (input === '/mcp-status' && mcpManager) { console.log(JSON.stringify(mcpManager.getStatus(), null, 2)); continue; }
      if (input === '/mcp-tools' && mcpManager) {
        const tools = mcpManager.getAllTools();
        console.log(`\n🔧 MCP 工具 (${tools.length} 个):\n`);
        for (const t of tools) console.log(`  [${t._mcpServer}] ${t.name}: ${(t.description || '').replace(/\[MCP:[^\]]*\]\s*/, '').slice(0, 80)}`);
        console.log(''); continue;
      }
      if (input.startsWith('/mcp-restart ') && mcpManager) {
        const name = input.slice('/mcp-restart '.length).trim();
        try { await mcpManager.restartServer(name); agent.refreshTools(); console.log(`✅ ${name} 已重启\n`); }
        catch (e) { console.log(`❌ ${e.message}\n`); }
        continue;
      }

      // LLM 路由状态
      if (input === '/llm-status' && llmRouter) {
        console.log('\n🧠 LLM 路由状态:\n');
        console.log(JSON.stringify(llmRouter.getStatus(), null, 2));
        console.log('');
        continue;
      }
      if (input === '/routes' && llmRouter) {
        console.log('\n🛤️  LLM 角色映射:');
        console.log(`   chat (默认):  ${llmRouter.chat.name}/${llmRouter.chat.model}`);
        console.log(`   router (路由): ${llmRouter.router.name}/${llmRouter.router.model}`);
        console.log(`   planner (规划): ${llmRouter.planner.name}/${llmRouter.planner.model}`);
        console.log(`   profile (画像): ${llmRouter.profile.name}/${llmRouter.profile.model}`);
        console.log('');
        continue;
      }

      const result = await agent.run(input);
      console.log(`\n🤖 ${result.content || result.error}\n`);
      if (result.toolCalls > 0) {
        console.log(`   (${result.toolCalls} 次工具, ${result.rounds} 轮`
          + (result.mcpCalls ? `, ${result.mcpCalls} 次 MCP` : '')
          + (result.activeSkill ? `, 🎯 ${result.activeSkill}` : '')
          + ')\n');
      }
    } catch (err) { console.error(`\n❌ ${err.message}\n`); }
  }

  rl.close();
}

// ============================================================
// 10. 主入口
// ============================================================

async function main() {
  const args = parseArgs(process.argv);

  if (args.help) { showHelp(); process.exit(0); }
  if (args.version) { showVersion(); process.exit(0); }
  if (args.listModels) { listModels(args.provider); process.exit(0); }

  // 加载配置（含 workspace/config.json）
  const config = ConfigManager.load({
    ...(args.provider && { provider: args.provider }),
    ...(args.model && { model: args.model }),
    ...(args.apiKey && { apiKey: args.apiKey }),
    ...(args.baseUrl && { baseUrl: args.baseUrl }),
    ...(args.workDir && { workDir: args.workDir }),
    ...(args.noMcp && { mcpEnabled: false }),
    ...(args.noSkills && { skillsEnabled: false })
  });

  // ========== Skills 管理（独立于 Agent） ==========
  const skillManager = config.skillsEnabled
    ? new SkillManager({ skillsDir: SKILLS_DIR, verbose: config.verbose })
    : null;

  // --skill-create
  if (args.skillCreate) {
    if (!skillManager) { console.error('❌ Skills 已禁用'); process.exit(1); }
    try {
      const skill = skillManager.create(args.skillCreate);
      console.log(`\n✅ 技能创建成功: ${skill.name}`);
      console.log(`   目录: ${skill.skillDir}`);
      console.log(`   请编辑 ${path.join(skill.skillDir, 'SKILL.md')} 补充技能内容`);
    } catch (err) { console.error(`❌ ${err.message}`); process.exit(1); }
    process.exit(0);
  }

  // --skill-list
  if (args.skillList) {
    if (!skillManager) { console.log('Skills 已禁用'); process.exit(0); }
    skillManager.scan();
    const skills = skillManager.list();
    if (skills.length === 0) { console.log('\n📭 暂无技能。用 --skill-create <name> 创建'); process.exit(0); }
    console.log(`\n🎯 可用技能 (${skills.length} 个):\n`);
    for (const s of skills) {
      console.log(`  [${s.category}] ${s.name}`);
      console.log(`    描述: ${s.description}`);
      console.log(`    触发词: ${s.triggerCount} 个 | LLM: ${s.llmRole || '默认'}\n`);
    }
    process.exit(0);
  }

  // --skill-remove
  if (args.skillRemove) {
    if (!skillManager) { console.error('❌ Skills 已禁用'); process.exit(1); }
    skillManager.scan();
    try { skillManager.remove(args.skillRemove); console.log(`✅ 已删除: ${args.skillRemove}`); }
    catch (err) { console.error(`❌ ${err.message}`); process.exit(1); }
    process.exit(0);
  }

  // ========== MCP 管理 ==========
  const mcpManager = config.mcpEnabled
    ? new MCPClientManager({ configPath: args.mcpConfig || undefined, verbose: config.verbose })
    : null;

  if (args.mcpAdd) {
    if (!mcpManager) { console.error('❌ MCP 已禁用'); process.exit(1); }
    const currentConfig = mcpManager.loadConfig();
    currentConfig[args.mcpAdd.name] = { url: args.mcpAdd.url };
    mcpManager.saveConfig(currentConfig);
    console.log(`✅ MCP Server 已添加: ${args.mcpAdd.name}\n   URL: ${args.mcpAdd.url}\n   配置: ${mcpManager.configPath}`);
    process.exit(0);
  }

  if (args.mcpRemove) {
    if (!mcpManager) { console.error('❌ MCP 已禁用'); process.exit(1); }
    const c = mcpManager.loadConfig(); delete c[args.mcpRemove]; mcpManager.saveConfig(c);
    console.log(`✅ 已移除: ${args.mcpRemove}`); process.exit(0);
  }

  if (args.mcpStatus) {
    if (!mcpManager) { console.log('MCP 已禁用'); process.exit(0); }
    try { await mcpManager.connectAll(); } catch { /* */ }
    console.log(JSON.stringify(mcpManager.getStatus(), null, 2)); process.exit(0);
  }

  if (args.mcpListTools) {
    if (!mcpManager) { console.log('MCP 已禁用'); process.exit(0); }
    try { await mcpManager.connectAll(); } catch { /* */ }
    const tools = mcpManager.getAllTools();
    console.log(`\n🔧 MCP 工具 (${tools.length} 个):\n`);
    for (const t of tools) {
      console.log(`  [${t._mcpServer}] ${t.name}`);
      console.log(`    描述: ${(t.description || '').slice(0, 100)}`);
      console.log(`    参数: ${Object.keys(t.inputSchema?.properties || {}).join(', ') || '(无)'}\n`);
    }
    process.exit(0);
  }

  if (args.saveConfig) {
    ConfigManager.save(config);
    console.log(`✅ 配置已保存: ${WORKSPACE_CONFIG_PATH}`);
    if (!args.prompt) process.exit(0);
  }

  // ========== 检查 API Key ==========
  const hasApiKey = config.llm?.api_key || config.apiKey;
  if (!hasApiKey) {
    console.error('❌ 缺少 API Key！');
    console.error('   方式 1: 编辑 workspace/config.json 设置 llm.api_key');
    console.error('   方式 2: export PI_API_KEY=sk-xxx');
    console.error('   方式 3: node pi-mvp.js --api-key sk-xxx "你好"');
    console.error('   方式 4: 本地 Ollama');
    console.error('');
    console.error('配置模板:');
    console.error(JSON.stringify({
      llm: { base_url: 'https://api.deepseek.com', api_key: 'sk-xxx', model: 'deepseek-chat' }
    }, null, 2));
    process.exit(1);
  }

  // ========== 初始化核心组件 ==========
  // 1. LLM Router（多 Provider 路由）
  const llmRouter = ConfigManager.createLLMRouter(config);

  // 2. 兼容旧版：如果只有单一 llm 配置，也创建一个普通 Provider
  const legacyProvider = ConfigManager.createProvider({
    provider: config.provider || 'openai',
    apiKey: config.llm?.api_key || config.apiKey,
    baseUrl: config.llm?.base_url || config.baseUrl,
    model: config.llm?.model || config.model || 'gpt-4o-mini'
  });

  // 3. 扫描 Skills（仅元数据）
  if (skillManager) {
    skillManager.router = llmRouter;  // 注入 LLM Router 用于意图路由
    skillManager.scan();
  }

  // 4. 连接 MCP
  if (mcpManager) {
    try { await mcpManager.connectAll(); } catch (err) {
      if (config.verbose) console.log(`⚠️  MCP: ${err.message}`);
    }
  }

  // 5. 创建 Agent
  const agent = new Agent({
    llmRouter,
    provider: legacyProvider,
    mcpManager,
    skillManager,
    workDir: config.workDir,
    maxIterations: config.agent?.max_rounds || config.maxIterations || 30,
    verbose: config.verbose
  });

  // 启动信息
  if (config.verbose) {
    console.log(`\n🔧 Pi-MVP v0.4 启动`);
    console.log(`   Agent: ${config.agent?.name || 'Pi-Agent'}`);
    console.log(`   LLM: ${legacyProvider.name}/${legacyProvider.model}`);
    if (llmRouter.roles.length > 1) {
      console.log(`   Routes: ${llmRouter.roles.join(', ')}`);
    }
    console.log(`   WorkDir: ${config.workDir}`);
    if (skillManager) console.log(`   Skills: ${skillManager.count} 个可用`);
    if (mcpManager) console.log(`   MCP: ${mcpManager.serverCount} Server, ${mcpManager.totalToolCount} tools`);
  }

  // ========== 运行 ==========
  if (args.prompt) {
    try {
      const result = await agent.run(args.prompt);
      console.log(result.content || result.error || '(无回复)');
      if (config.verbose && result.toolCalls > 0) {
        console.log(`\n📊 ${result.toolCalls} 次 | ${result.rounds} 轮`
          + (result.mcpCalls ? ` | MCP: ${result.mcpCalls}` : '')
          + (result.activeSkill ? ` | 🎯 ${result.activeSkill}` : ''));
      }
    } catch (err) { console.error(`❌ ${err.message}`); process.exit(1); }
  } else {
    await startRepl(agent, mcpManager, skillManager, llmRouter);
  }

  // 清理
  if (mcpManager) mcpManager.disconnectAll();
}

module.exports = {
  BaseProvider, OpenAICompatible, AnthropicProvider, GoogleProvider,
  LLMRouter,
  Tool, BuiltinTools, getAllBuiltinTools, getBuiltinToolByName,
  buildMergedTools, findToolByName, mcpToolToAgentTool,
  MCPProtocol, MCPHttpTransport, MCPServer, MCPClientManager,
  Skill, SkillManager,
  Session, Agent, ConfigManager, DEFAULT_SYSTEM_PROMPT, parseArgs, main, Utils,
  WORKSPACE_DIR, SKILLS_DIR, WORKSPACE_CONFIG_PATH
};

if (require.main === module) {
  main().catch(err => { console.error('💥 致命错误:', err.message); process.exit(1); });
}
