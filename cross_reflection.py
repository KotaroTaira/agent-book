from reflection_manager import ReflectionManager, TaskReflector
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from self_reflection import ReflectiveAgent

def main():
    import argparse

    from settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
      description="ReflectiveAgentを使用してタスクを実行します（Cross-Reflection）"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    openai_llm = ChatOpenAI(model=settings.openai_smart_model, temperature=settings.temperature)
    anthropic_llm = ChatAnthropic(model=settings.anthropic_model, temperature=settings.temperature)

    reflection_manager = ReflectionManager(file_path=settings.default_reflection_db_path)
    task_reflector = TaskReflector(llm=anthropic_llm, reflection_manager=reflection_manager)
    agent = ReflectiveAgent(
        llm=openai_llm,
        reflection_manager=reflection_manager,
        task_reflector=task_reflector,
    )
    result = agent.run(args.task)
    print(result)

if __name__ == "__main__":
    main()
