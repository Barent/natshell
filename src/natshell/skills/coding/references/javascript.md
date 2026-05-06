# JavaScript / TypeScript reference

## Project layout signals
- `package.json` → project root
- `tsconfig.json` → TypeScript project
- `src/` for source, `dist/` or `build/` for output

## Run tests
```bash
npm test                   # delegates to test script in package.json
npx jest                   # Jest directly
npx vitest run             # Vitest
npx mocha                  # Mocha
```

## Build / check
```bash
npm run build              # tsc or bundler
npx tsc --noEmit           # type-check only (TypeScript)
npx eslint src/            # lint (if configured)
```

## ESM vs CommonJS
- `"type": "module"` in package.json → ESM (`import`/`export`)
- Otherwise → CommonJS (`require`/`module.exports`)
- In ESM, `__dirname` is not available — use `import.meta.url` + `fileURLToPath`

## Common idioms
- Prefer `const` over `let`; avoid `var`
- Use optional chaining `?.` and nullish coalescing `??`
- `async/await` over raw Promise chains
- TypeScript: use `unknown` over `any`; narrow with type guards

## Gotchas
- `==` vs `===`: always use `===`
- `Array.forEach` cannot be broken out of; use `for...of` for break/continue
- `JSON.parse` returns `any` — validate before use
- Node `require` cache: changes to required modules require process restart
