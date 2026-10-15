// Host-injected text inside a summarized turn. Text that carries the
// conversation forward stays: a user-run tool, a subagent task result, a
// working-directory change and a user comment. Everything else is transient
// host reporting (file expansions, reminders, progress) that the summary
// replaces, and is dropped. Each host applies this rule to the records it
// stores as injected text.
export function keepsHostInjectedText(text: string): boolean {
  return (
    text === "The following tool was executed by the user"
    || text.startsWith("<task ")
    || text.includes("The user has changed the current working directory to")
    || text.startsWith("The user made the following comment")
  );
}
