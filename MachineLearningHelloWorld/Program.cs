using System.Reflection;
using System.Threading.Tasks;
using Oakton;

namespace MachineLearningHelloWorld
{
    class Program
    {
        private static async Task<int> Main(string[] args)
        {
            var executor = CommandExecutor.For(_ =>
            {
                _.RegisterCommands(typeof(Program).GetTypeInfo().Assembly);
            });

            return await executor.ExecuteAsync(args);
        }
    }
}
